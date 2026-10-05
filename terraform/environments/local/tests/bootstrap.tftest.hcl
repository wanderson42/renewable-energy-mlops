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
    condition     = length(helm_release.mlops.values) == 3 && strcontains(nonsensitive(helm_release.mlops.values[1]), "sha256:6af326440af64565c2ea5d1d28c9ee355f70a8d4af5b76e35938365e1ce2e428")
    error_message = "Bootstrap must load the pinned image profile after the credentials."
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
