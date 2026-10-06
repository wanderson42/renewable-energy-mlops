variable "project" {
  type    = string
  default = "energy-mlops"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,19}$", var.project))
    error_message = "Use 3–20 lowercase letters, digits or hyphens, starting with a letter."
  }
}

variable "environment" {
  type    = string
  default = "portfolio"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,9}$", var.environment))
    error_message = "Use 2–10 lowercase letters, digits or hyphens, starting with a letter."
  }
}

variable "account_id" {
  description = "Target AWS account; also restricts the provider. Tests use a fictional account."
  type        = string
  validation {
    condition     = can(regex("^[0-9]{12}$", var.account_id))
    error_message = "Provide a twelve-digit account ID."
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "availability_zones" {
  description = "Exactly two distinct AZs in the selected region; verify availability before real cloud use."
  type        = list(string)
  validation {
    condition     = length(var.availability_zones) == 2 && length(toset(var.availability_zones)) == 2 && alltrue([for az in var.availability_zones : startswith(az, var.region)])
    error_message = "Provide two distinct AZs belonging to the selected region."
  }
}

variable "vpc_cidr" {
  type    = string
  default = "10.42.0.0/16"
  validation {
    condition     = can(cidrnetmask(var.vpc_cidr)) && try(tonumber(split("/", var.vpc_cidr)[1]) >= 16 && tonumber(split("/", var.vpc_cidr)[1]) <= 20, false)
    error_message = "Use an IPv4 CIDR with prefix /16–/20; derived subnets must remain within AWS sizes."
  }
}

variable "kubernetes_version" {
  description = "Explicit version supported by EKS in the target region; no inheritance from KinD."
  type        = string
  validation {
    condition     = can(regex("^1\\.[0-9]{2}$", var.kubernetes_version))
    error_message = "Provide an explicit Kubernetes minor version, such as 1.xx."
  }
}

variable "node_ami_release" {
  description = "Explicit EKS AL2023 AMI release compatible with kubernetes_version; validate against AWS before use."
  type        = string
  validation {
    condition     = length(trimspace(var.node_ami_release)) > 0
    error_message = "Pin an AMI release rather than resolving latest during deployment."
  }
}

variable "addon_versions" {
  description = "Explicit EKS add-on versions compatible with the selected Kubernetes version."
  type = object({
    vpc_cni    = string
    coredns    = string
    kube_proxy = string
  })
  validation {
    condition     = alltrue([for version in values(var.addon_versions) : can(regex("^v[0-9]+\\.[0-9]+\\.[0-9]+-eksbuild\\.[0-9]+$", version))])
    error_message = "Pin each add-on using the vX.Y.Z-eksbuild.N format."
  }
}

variable "admin_role_arn" {
  description = "Existing administrative role in the target account; no implicit creator admin access."
  type        = string
  validation {
    condition     = can(regex("^arn:aws:iam::${var.account_id}:role/.+$", var.admin_role_arn))
    error_message = "Provide a role ARN from the target account."
  }
}

variable "public_api_cidrs" {
  description = "Optional restricted IPv4 ranges for the Kubernetes API; empty means private-only."
  type        = list(string)
  default     = []
  validation {
    condition     = alltrue([for cidr in var.public_api_cidrs : can(cidrnetmask(cidr)) && try(tonumber(split("/", cidr)[1]) >= 24, false)])
    error_message = "Use restricted IPv4 CIDRs (/24 or narrower); unrestricted API access is rejected."
  }
}

variable "db_engine_version" {
  description = "Explicit RDS PostgreSQL version available in the selected region."
  type        = string
  validation {
    condition     = can(regex("^[0-9]+\\.[0-9]+$", var.db_engine_version))
    error_message = "Provide a PostgreSQL engine version in major.minor format."
  }
}

variable "db_final_snapshot_identifier" {
  description = "Reserved unique name for an eventual final RDS snapshot; no destroy is automated."
  type        = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,62}$", var.db_final_snapshot_identifier)) && !strcontains(var.db_final_snapshot_identifier, "--") && !endswith(var.db_final_snapshot_identifier, "-")
    error_message = "Provide a valid explicit final-snapshot identifier."
  }
}

variable "workload_namespace" {
  type    = string
  default = "energy-mlops"
  validation {
    condition     = can(regex("^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$", var.workload_namespace))
    error_message = "Use a valid Kubernetes namespace name."
  }
}
