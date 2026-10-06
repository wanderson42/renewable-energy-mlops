resource "aws_iam_role" "cluster" {
  name = "${local.name}-cluster"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "eks.amazonaws.com" } }]
  })
}

resource "aws_iam_role_policy_attachment" "cluster" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_iam_role" "nodes" {
  name = "${local.name}-nodes"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "ec2.amazonaws.com" } }]
  })
}

resource "aws_iam_role_policy_attachment" "nodes" {
  for_each   = toset(["AmazonEKSWorkerNodePolicy", "AmazonEC2ContainerRegistryPullOnly"])
  role       = aws_iam_role.nodes.name
  policy_arn = "arn:aws:iam::aws:policy/${each.key}"
}

resource "aws_iam_openid_connect_provider" "eks" {
  url            = aws_eks_cluster.mlops.identity[0].oidc[0].issuer
  client_id_list = ["sts.amazonaws.com"]
  # AWS retrieves the thumbprint when omitted; do not embed an expired hash.
}

locals {
  oidc_host = trimprefix(aws_iam_openid_connect_provider.eks.url, "https://")
  service_accounts = {
    mlflow    = "mlflow"
    api       = "energy-api"
    pipelines = "energy-pipelines"
  }
  workload_buckets = {
    mlflow    = ["mlflow-artifacts"]
    api       = ["energy-lake", "mlflow-artifacts"]
    pipelines = ["energy-lake", "mlflow-artifacts"]
  }
}

resource "aws_iam_role" "cni" {
  name = "${local.name}-cni"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRoleWithWebIdentity"
      Principal = { Federated = aws_iam_openid_connect_provider.eks.arn }
      Condition = { StringEquals = {
        "${local.oidc_host}:aud" = "sts.amazonaws.com"
        "${local.oidc_host}:sub" = "system:serviceaccount:kube-system:aws-node"
      } }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "cni" {
  role       = aws_iam_role.cni.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role" "workload" {
  for_each = local.service_accounts
  name     = "${local.name}-${each.key}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRoleWithWebIdentity"
      Principal = { Federated = aws_iam_openid_connect_provider.eks.arn }
      Condition = { StringEquals = {
        "${local.oidc_host}:aud" = "sts.amazonaws.com"
        "${local.oidc_host}:sub" = "system:serviceaccount:${var.workload_namespace}:${each.value}"
      } }
    }]
  })
}

resource "aws_iam_role_policy" "storage" {
  for_each = local.workload_buckets
  name     = "scoped-s3"
  role     = aws_iam_role.workload[each.key].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = [for key in each.value : aws_s3_bucket.data[key].arn]
      },
      {
        Effect   = "Allow"
        Action   = each.key == "api" ? ["s3:GetObject", "s3:GetObjectVersion"] : ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"]
        Resource = [for key in each.value : "${aws_s3_bucket.data[key].arn}/*"]
      }
    ]
  })
}

# Materialization/rotation of this secret into the MLflow pod is a deployment
# integration task, not implemented by granting IAM permissions alone.
resource "aws_iam_role_policy" "database_secret" {
  name = "mlflow-database-secret"
  role = aws_iam_role.workload["mlflow"].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [aws_db_instance.mlflow.master_user_secret[0].secret_arn]
    }]
  })
}
