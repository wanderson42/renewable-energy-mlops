terraform {
  required_version = "= 1.16.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "= 6.67.0"
    }
  }

  # Separate from the executable KinD environment. Cloud use would require a
  # reviewed remote backend with locking and access controls before any apply.
  backend "local" {
    path = "../../../.repro/aws.tfstate"
  }
}

provider "aws" {
  region              = var.region
  allowed_account_ids = [var.account_id]

  default_tags {
    tags = {
      Project     = var.project
      Environment = var.environment
      ManagedBy   = "Terraform"
      Purpose     = "PortfolioBlueprint"
    }
  }
}
