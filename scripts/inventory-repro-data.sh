#!/usr/bin/env bash
set -euo pipefail
umask 077

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v kubectl >/dev/null || { echo "kubectl ausente." >&2; exit 1; }
[[ -f "$project_root/.repro/kubeconfig" ]] || { echo "Kubeconfig de ensaio ausente." >&2; exit 1; }

output_dir="$(mktemp -d "$project_root/.repro/inventory.XXXXXX")"
trap 'rm -rf -- "$output_dir"' EXIT

kubectl --context kind-energy-mlops --namespace default \
  exec -i deployment/mlflow -- python3 - --role source \
  < "$project_root/scripts/inventory-repro-data.py" > "$output_dir/source.json"

kubectl --kubeconfig "$project_root/.repro/kubeconfig" \
  --context kind-energy-mlops-repro --namespace energy-mlops-repro \
  exec -i deployment/mlflow -- python3 - --role destination \
  < "$project_root/scripts/inventory-repro-data.py" > "$output_dir/destination.json"

mv -- "$output_dir/source.json" "$project_root/.repro/source-data-inventory.json"
mv -- "$output_dir/destination.json" "$project_root/.repro/destination-data-inventory.json"
cat "$project_root/.repro/source-data-inventory.json"
cat "$project_root/.repro/destination-data-inventory.json"

kubectl --kubeconfig "$project_root/.repro/kubeconfig" \
  --context kind-energy-mlops-repro --namespace energy-mlops-repro \
  get pods -l 'app in (mlflow,postgres,rustfs)' \
  -o 'custom-columns=NAME:.metadata.name,IMAGE_IDS:.status.containerStatuses[*].imageID,READY:.status.containerStatuses[*].ready'
