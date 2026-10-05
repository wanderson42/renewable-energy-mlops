locals {
  project_root = abspath("${path.module}/../../..")
  chart_path   = "${local.project_root}/helm"
  context      = "kind-energy-mlops-repro"
  namespace    = "energy-mlops-repro"
  release      = "energy-mlops-repro"
  # Detect local chart edits even when the chart directory path is unchanged.
  chart_hash = sha256(join("", [
    for name in sort(tolist(fileset(local.chart_path, "**"))) :
    filesha256("${local.chart_path}/${name}")
    if name == "Chart.yaml" || name == "values.yaml" || startswith(name, "templates/")
  ]))
}

provider "helm" {
  kubernetes = {
    config_path    = "${local.project_root}/.repro/kubeconfig"
    config_context = local.context
  }
}

resource "helm_release" "mlops" {
  name             = local.release
  namespace        = local.namespace
  create_namespace = true
  chart            = local.chart_path
  wait             = true
  timeout          = var.deployment_timeout_seconds
  # Retain failed installs for diagnosis; do not delete data on a timeout.
  atomic          = false
  upgrade_install = false
  reset_values    = true

  values = [
    sensitive(try(file(var.secrets_file), "")),
    file("${local.chart_path}/environments/repro.yaml"),
    file("${local.chart_path}/environments/repro-bootstrap.yaml"),
  ]

  # Overrides are applied last: credentials cannot change the chosen stage.
  set = [
    { name = "api.enabled", value = tostring(var.api_enabled) },
    { name = "api.image.digest", value = var.api_digest, type = "string" },
    { name = "repro.chartHash", value = local.chart_hash, type = "string" },
  ]
}
