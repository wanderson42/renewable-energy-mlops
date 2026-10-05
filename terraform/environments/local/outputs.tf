output "deployment_target" {
  description = "Fixed rehearsal target and the explicitly selected serving stage."
  value = {
    context     = local.context
    namespace   = helm_release.mlops.namespace
    release     = helm_release.mlops.name
    api_enabled = var.api_enabled
  }
}
