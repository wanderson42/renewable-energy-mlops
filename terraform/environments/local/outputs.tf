output "deployment_target" {
  description = "Fixed bootstrap target; serving is disabled until restoration."
  value = {
    context     = local.context
    namespace   = helm_release.mlops.namespace
    release     = helm_release.mlops.name
    api_enabled = false
  }
}
