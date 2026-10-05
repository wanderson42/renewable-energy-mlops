#!/usr/bin/env bash
set -euo pipefail
umask 077

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
terraform_bin="$project_root/.repro/bin/terraform"
terraform_root="$project_root/terraform/environments/local"
kubeconfig="$project_root/.repro/kubeconfig"

[[ -x "$terraform_bin" ]] || { echo "Execute make repro-tools primeiro." >&2; exit 1; }
[[ -f "$kubeconfig" ]] || { echo "Kubeconfig de ensaio ausente; execute make repro-cluster." >&2; exit 1; }
[[ -f "$project_root/helm/values_secrets.yaml" ]] || { echo "Arquivo local helm/values_secrets.yaml ausente." >&2; exit 1; }
command -v kubectl >/dev/null || { echo "kubectl ausente." >&2; exit 1; }

kubectl --kubeconfig "$kubeconfig" --context kind-energy-mlops-repro get nodes
"$terraform_bin" -chdir="$terraform_root" fmt -check -recursive
"$terraform_bin" -chdir="$terraform_root" init -input=false -lockfile=readonly
"$terraform_bin" -chdir="$terraform_root" validate
"$terraform_bin" -chdir="$terraform_root" plan -input=false \
  -out="$project_root/.repro/bootstrap.tfplan"
echo "Plano salvo em .repro/bootstrap.tfplan. Nenhum workload foi aplicado."
