variable "api_enabled" {
  description = "Enable serving only after restoring and verifying the rehearsal data."
  type        = bool
  default     = false
}

variable "api_digest" {
  description = "Immutable API image reference; the default is the validated v17 baseline."
  type        = string
  default     = "sha256:95208ab282e24014a81f60d1e3eac01b8db36e40f05ae25e9f168264e4b7c188"
  validation {
    condition     = can(regex("^sha256:[a-f0-9]{64}$", var.api_digest))
    error_message = "Use a complete lowercase sha256 digest."
  }
}

variable "deployment_timeout_seconds" {
  description = "Helm wait deadline; the failure rehearsal explicitly uses 60 seconds."
  type        = number
  default     = 600
  validation {
    condition     = var.deployment_timeout_seconds >= 60 && var.deployment_timeout_seconds <= 900 && floor(var.deployment_timeout_seconds) == var.deployment_timeout_seconds
    error_message = "Use an integer wait deadline between 60 and 900 seconds."
  }
}

variable "secrets_file" {
  description = "Local Helm credentials file; never commit its contents or state."
  type        = string
  default     = "../../../helm/values_secrets.yaml"

  validation {
    condition     = fileexists(var.secrets_file)
    error_message = "Provide an existing local Helm credentials file."
  }

  validation {
    condition = alltrue([
      for field in [
        ["postgres", "user"], ["postgres", "password"], ["postgres", "db"],
        ["rustfs", "rootUser"], ["rustfs", "rootPassword"],
      ] : try(length(trimspace(yamldecode(file(var.secrets_file))[field[0]][field[1]])) > 0, false)
    ])
    error_message = "The credentials file must contain nonempty postgres user/password/db and RustFS rootUser/rootPassword."
  }
}
