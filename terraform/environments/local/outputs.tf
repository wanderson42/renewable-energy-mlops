output "deployment_target" {
  description = "Fixed rehearsal target and the explicitly selected serving stage."
  value = {
    context                         = local.context
    namespace                       = helm_release.mlops.namespace
    release                         = helm_release.mlops.name
    api_enabled                     = var.api_enabled
    api_digest                      = var.api_digest
    deployment_timeout_seconds      = var.deployment_timeout_seconds
    mlflow_image                    = "${local.mlflow_image.image.repository}@${local.mlflow_image.image.digest}"
    mlflow_install_runtime_packages = local.mlflow_image.installRuntimePackages
  }
}
