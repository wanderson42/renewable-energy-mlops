mock_provider "helm" {}

variables {
  secrets_file = "tests/fixtures/credentials.yaml"
}

run "isolated_bootstrap" {
  command = plan

  assert {
    condition     = helm_release.mlops.name == "energy-mlops-repro" && helm_release.mlops.namespace == "energy-mlops-repro"
    error_message = "Bootstrap must use the dedicated reproduction release and namespace."
  }

  assert {
    condition     = one([for setting in helm_release.mlops.set : setting.value if setting.name == "api.enabled"]) == "false"
    error_message = "Serving must remain disabled even when a credentials file enables it."
  }

  assert {
    condition     = helm_release.mlops.wait && !helm_release.mlops.atomic && !helm_release.mlops.upgrade_install
    error_message = "Wait for infrastructure, retain failures, and never take over an existing unmanaged release."
  }

  assert {
    condition     = length(helm_release.mlops.values) == 3 && strcontains(nonsensitive(helm_release.mlops.values[1]), "sha256:233063b1cf82a2fb72426a2b09334ef31fb73b1db4bf7d1d5cfb8d4b4c0e657f") && !output.deployment_target.mlflow_install_runtime_packages
    error_message = "Bootstrap must load the pinned image profile after the credentials."
  }
}

run "packaged_mlflow_runtime" {
  command = plan
  assert {
    condition     = output.deployment_target.mlflow_image == "ghcr.io/wanderson42/renewable-energy-mlops@sha256:233063b1cf82a2fb72426a2b09334ef31fb73b1db4bf7d1d5cfb8d4b4c0e657f"
    error_message = "The rehearsal must select the published MLflow runtime digest."
  }
}

run "missing_credentials_rejected" {
  command = plan
  variables {
    secrets_file = "tests/fixtures/missing.yaml"
  }
  expect_failures = [var.secrets_file]
}

run "serving_enabled_explicitly" {
  command = plan
  variables {
    api_enabled = true
  }
  assert {
    condition     = one([for setting in helm_release.mlops.set : setting.value if setting.name == "api.enabled"]) == "true" && output.deployment_target.api_enabled
    error_message = "The serving stage must enable the API in the same isolated release."
  }
}

run "controlled_pull_failure_configuration" {
  command = plan
  variables {
    api_enabled                = true
    api_digest                 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    deployment_timeout_seconds = 60
  }
  assert {
    condition     = helm_release.mlops.timeout == 60 && one([for setting in helm_release.mlops.set : setting.value if setting.name == "api.image.digest"]) == var.api_digest && output.deployment_target.api_digest == var.api_digest
    error_message = "The controlled failure must use the selected digest and bounded wait deadline."
  }
}

run "incomplete_digest_rejected" {
  command = plan
  variables {
    api_digest = "sha256:incomplete"
  }
  expect_failures = [var.api_digest]
}
