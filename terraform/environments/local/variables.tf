variable "api_enabled" {
  description = "Enable serving only after restoring and verifying the rehearsal data."
  type        = bool
  default     = false
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
