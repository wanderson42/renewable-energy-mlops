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
target_nodes="$(kubectl --kubeconfig "$kubeconfig" --context kind-energy-mlops-repro get nodes -o 'jsonpath={.items[*].metadata.name}')"
[[ "$target_nodes" == energy-mlops-repro-control-plane ]] || { echo "Node do ensaio não confirmado." >&2; exit 1; }
plan_args=()
plan_file="$project_root/.repro/bootstrap.tfplan"
if [[ -f "$project_root/.repro/deployment.tfvars.json" ]]; then
  plan_args+=("-var-file=$project_root/.repro/deployment.tfvars.json")
fi
if [[ "${1:-}" == --serving ]]; then
  plan_args+=("-var=api_enabled=true")
  plan_file="$project_root/.repro/serving.tfplan"
elif [[ $# -gt 0 ]]; then
  echo "Argumento desconhecido." >&2; exit 1
fi
"$terraform_bin" -chdir="$terraform_root" fmt -check -recursive
"$terraform_bin" -chdir="$terraform_root" init -input=false -lockfile=readonly
"$terraform_bin" -chdir="$terraform_root" validate
"$terraform_bin" -chdir="$terraform_root" plan -input=false \
  "${plan_args[@]}" -out="$plan_file"
echo "Plano salvo em $plan_file. Nenhum workload foi aplicado."
