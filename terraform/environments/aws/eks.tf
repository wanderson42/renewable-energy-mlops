resource "aws_eks_cluster" "mlops" {
  bootstrap_self_managed_addons = false
  deletion_protection           = true
  name                          = local.name
  role_arn                      = aws_iam_role.cluster.arn
  version                       = var.kubernetes_version
  enabled_cluster_log_types     = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  access_config {
    authentication_mode                         = "API"
    bootstrap_cluster_creator_admin_permissions = false
  }

  vpc_config {
    subnet_ids              = [for subnet in aws_subnet.workloads : subnet.id]
    endpoint_private_access = true
    endpoint_public_access  = length(var.public_api_cidrs) > 0
    public_access_cidrs     = length(var.public_api_cidrs) > 0 ? var.public_api_cidrs : null
  }

  depends_on = [aws_iam_role_policy_attachment.cluster, aws_cloudwatch_log_group.eks]
}

resource "aws_cloudwatch_log_group" "eks" {
  name              = "/aws/eks/${local.name}/cluster"
  retention_in_days = 30
}

resource "aws_eks_access_entry" "admin" {
  cluster_name  = aws_eks_cluster.mlops.name
  principal_arn = var.admin_role_arn
  type          = "STANDARD"
}

resource "aws_eks_access_policy_association" "admin" {
  cluster_name  = aws_eks_cluster.mlops.name
  principal_arn = aws_eks_access_entry.admin.principal_arn
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
  access_scope { type = "cluster" }
}

resource "aws_launch_template" "nodes" {
  name_prefix = "${local.name}-"
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 40
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }
}

resource "aws_eks_node_group" "workloads" {
  cluster_name    = aws_eks_cluster.mlops.name
  node_group_name = "workloads"
  node_role_arn   = aws_iam_role.nodes.arn
  subnet_ids      = [for subnet in aws_subnet.workloads : subnet.id]
  version         = var.kubernetes_version
  release_version = var.node_ami_release
  ami_type        = "AL2023_x86_64_STANDARD"
  capacity_type   = "ON_DEMAND"
  instance_types  = ["t3.large"]
  scaling_config {
    desired_size = 2
    min_size     = 2
    max_size     = 3
  }
  update_config { max_unavailable = 1 }
  launch_template {
    id      = aws_launch_template.nodes.id
    version = tostring(aws_launch_template.nodes.latest_version)
  }
  depends_on = [aws_iam_role_policy_attachment.nodes, aws_eks_addon.vpc_cni, aws_route.workload_egress]
}

resource "aws_eks_addon" "vpc_cni" {
  cluster_name             = aws_eks_cluster.mlops.name
  addon_name               = "vpc-cni"
  addon_version            = var.addon_versions.vpc_cni
  service_account_role_arn = aws_iam_role.cni.arn
  depends_on               = [aws_iam_role_policy_attachment.cni]
}

resource "aws_eks_addon" "core" {
  for_each = {
    coredns      = var.addon_versions.coredns
    "kube-proxy" = var.addon_versions.kube_proxy
  }
  cluster_name  = aws_eks_cluster.mlops.name
  addon_name    = each.key
  addon_version = each.value
  depends_on    = [aws_eks_node_group.workloads]
}
