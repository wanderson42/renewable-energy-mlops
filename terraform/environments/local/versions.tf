terraform {
  required_version = "= 1.16.5"

  required_providers {
    helm = {
      source  = "hashicorp/helm"
      version = "= 3.3.0"
    }
  }

  backend "local" {
    path = "../../../.repro/terraform.tfstate"
  }
}
