output "cloud_blueprint" {
  description = "Connection contract only. Outputs do not install workloads or restore model state."
  value = {
    region              = var.region
    cluster             = aws_eks_cluster.mlops.name
    namespace           = var.workload_namespace
    buckets             = { for key, bucket in aws_s3_bucket.data : key => bucket.bucket }
    database            = { host = aws_db_instance.mlflow.address, port = 5432, name = "mlflow", sslmode = "verify-full" }
    database_secret_arn = aws_db_instance.mlflow.master_user_secret[0].secret_arn
    service_accounts = {
      for key, name in local.service_accounts : key => {
        name     = name
        role_arn = aws_iam_role.workload[key].arn
      }
    }
    api_private_access                  = true
    api_public_access                   = length(var.public_api_cidrs) > 0
    applications_installed_by_this_root = false
  }
}
