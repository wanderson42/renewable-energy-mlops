# These inputs are fictional test fixtures, not deployable AWS recommendations.
mock_provider "aws" {
  override_during = plan

  mock_resource "aws_eks_cluster" {
    defaults = {
      arn        = "arn:aws:eks:us-east-1:123456789012:cluster/energy-mlops-portfolio"
      identity   = [{ oidc = [{ issuer = "https://oidc.eks.us-east-1.amazonaws.com/id/TESTONLY" }] }]
      vpc_config = { cluster_security_group_id = "sg-0123456789abcdef0" }
    }
  }
  mock_resource "aws_iam_openid_connect_provider" {
    defaults = {
      arn = "arn:aws:iam::123456789012:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/TESTONLY"
    }
  }
  mock_resource "aws_iam_role" {
    defaults = { arn = "arn:aws:iam::123456789012:role/test-only" }
  }
  mock_resource "aws_db_instance" {
    defaults = {
      master_user_secret = [{ secret_arn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:test-only" }]
    }
  }
  override_resource {
    target = aws_s3_bucket.data["energy-lake"]
    values = { arn = "arn:aws:s3:::energy-mlops-portfolio-123456789012-energy-lake" }
  }
  override_resource {
    target = aws_s3_bucket.data["mlflow-artifacts"]
    values = { arn = "arn:aws:s3:::energy-mlops-portfolio-123456789012-mlflow-artifacts" }
  }
}

variables {
  account_id                   = "123456789012"
  availability_zones           = ["us-east-1a", "us-east-1b"]
  admin_role_arn               = "arn:aws:iam::123456789012:role/test-only-admin"
  kubernetes_version           = "1.34"
  node_ami_release             = "1.34.0-20260101"
  db_engine_version            = "17.6"
  db_final_snapshot_identifier = "energy-mlops-test-only-final"
  addon_versions = {
    vpc_cni    = "v1.20.0-eksbuild.1"
    coredns    = "v1.12.0-eksbuild.1"
    kube_proxy = "v1.34.0-eksbuild.1"
  }
}

run "private_network_and_database" {
  command = plan
  assert {
    condition     = length(aws_subnet.workloads) == 2 && length(aws_subnet.database) == 2 && alltrue([for subnet in aws_subnet.workloads : !subnet.map_public_ip_on_launch]) && length(aws_nat_gateway.egress) == 2
    error_message = "Workloads must span two private subnets with AZ-local egress."
  }
  assert {
    condition     = length(aws_route_table.database.route) == 0 && !aws_db_instance.mlflow.publicly_accessible && aws_db_instance.mlflow.multi_az && aws_db_instance.mlflow.storage_encrypted && aws_db_instance.mlflow.manage_master_user_password
    error_message = "RDS must be isolated, encrypted and use managed credentials."
  }
  assert {
    condition     = aws_vpc_security_group_ingress_rule.postgres.from_port == 5432 && aws_vpc_security_group_ingress_rule.postgres.to_port == 5432 && aws_vpc_security_group_ingress_rule.postgres.cidr_ipv4 == null
    error_message = "PostgreSQL ingress must come from the EKS security group, not a CIDR."
  }
  assert {
    condition     = aws_db_instance.mlflow.deletion_protection && !aws_db_instance.mlflow.skip_final_snapshot && aws_db_instance.mlflow.backup_retention_period == 7
    error_message = "Database deletion and backup policy must remain explicit."
  }
}

run "private_eks_and_explicit_identity" {
  command = plan
  assert {
    condition     = aws_eks_cluster.mlops.vpc_config[0].endpoint_private_access && !aws_eks_cluster.mlops.vpc_config[0].endpoint_public_access && aws_eks_cluster.mlops.access_config[0].authentication_mode == "API" && !aws_eks_cluster.mlops.access_config[0].bootstrap_cluster_creator_admin_permissions
    error_message = "Cluster access must be private by default, with explicit admin identity."
  }
  assert {
    condition     = aws_launch_template.nodes.metadata_options[0].http_tokens == "required" && aws_launch_template.nodes.metadata_options[0].http_put_response_hop_limit == 1 && one(aws_launch_template.nodes.block_device_mappings).ebs[0].encrypted
    error_message = "Worker nodes must use IMDSv2 and encrypted disks."
  }
  assert {
    condition     = !aws_eks_cluster.mlops.bootstrap_self_managed_addons && aws_eks_node_group.workloads.release_version == var.node_ami_release && aws_eks_addon.vpc_cni.addon_version == var.addon_versions.vpc_cni && !output.cloud_blueprint.applications_installed_by_this_root
    error_message = "Pin node/add-on versions and distinguish infra from application deployment."
  }
}

run "storage_and_workload_permissions" {
  command = plan
  assert {
    condition     = alltrue([for config in aws_s3_bucket_public_access_block.data : config.block_public_acls && config.block_public_policy && config.ignore_public_acls && config.restrict_public_buckets]) && aws_s3_bucket_versioning.lake.versioning_configuration[0].status == "Enabled" && aws_s3_bucket_versioning.artifacts.versioning_configuration[0].status == "Enabled"
    error_message = "Both stores must block public access and preserve object versions."
  }
  assert {
    condition     = alltrue([for config in aws_s3_bucket_server_side_encryption_configuration.data : one(config.rule).apply_server_side_encryption_by_default[0].sse_algorithm == "AES256"])
    error_message = "Both buckets must explicitly enable encryption at rest."
  }
  assert {
    condition     = jsondecode(aws_s3_bucket_policy.tls["mlflow-artifacts"].policy).Statement[0].Condition.Bool["aws:SecureTransport"] == "false" && jsondecode(aws_s3_bucket_policy.tls["mlflow-artifacts"].policy).Statement[0].Effect == "Deny"
    error_message = "Storage must deny unencrypted transport."
  }
  assert {
    condition     = !contains(jsondecode(aws_iam_role_policy.storage["api"].policy).Statement[1].Action, "s3:PutObject") && !contains(jsondecode(aws_iam_role_policy.storage["api"].policy).Statement[1].Action, "s3:DeleteObject") && jsondecode(aws_iam_role_policy.storage["mlflow"].policy).Statement[0].Resource == [aws_s3_bucket.data["mlflow-artifacts"].arn]
    error_message = "API must be read-only and MLflow must be restricted to the artifact bucket."
  }
  assert {
    condition     = jsondecode(aws_iam_role.workload["api"].assume_role_policy).Statement[0].Condition.StringEquals["${local.oidc_host}:sub"] == "system:serviceaccount:energy-mlops:energy-api" && jsondecode(aws_iam_role.workload["api"].assume_role_policy).Statement[0].Condition.StringEquals["${local.oidc_host}:aud"] == "sts.amazonaws.com"
    error_message = "IRSA must bind the role to the exact namespace, service account and audience."
  }
}

run "restricted_public_api_opt_in" {
  command = plan
  variables { public_api_cidrs = ["192.0.2.10/32"] }
  assert {
    condition     = aws_eks_cluster.mlops.vpc_config[0].endpoint_public_access && aws_eks_cluster.mlops.vpc_config[0].public_access_cidrs == toset(["192.0.2.10/32"])
    error_message = "Public API access must require an explicit restricted IPv4 range."
  }
}

run "unrestricted_api_rejected" {
  command = plan
  variables { public_api_cidrs = ["0.0.0.0/0"] }
  expect_failures = [var.public_api_cidrs]
}

run "duplicate_zones_rejected" {
  command = plan
  variables { availability_zones = ["us-east-1a", "us-east-1a"] }
  expect_failures = [var.availability_zones]
}

run "wrong_account_admin_rejected" {
  command = plan
  variables { admin_role_arn = "arn:aws:iam::999999999999:role/wrong-account" }
  expect_failures = [var.admin_role_arn]
}
