#!/usr/bin/env bash
set -euo pipefail
umask 077

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for tool in kubectl python3 curl; do
  command -v "$tool" >/dev/null || { echo "Ferramenta ausente: $tool" >&2; exit 1; }
done
restore_phase=preflight
trap 'echo "Restauração interrompida na fase $restore_phase. Inspecione o ensaio; dados parciais foram preservados sem limpeza automática." >&2' ERR
[[ -f "$project_root/.repro/kubeconfig" && -f "$project_root/.repro/latest-snapshot" ]] || {
  echo "Kubeconfig ou referência do snapshot ausente." >&2; exit 1;
}
read -r snapshot_dir < "$project_root/.repro/latest-snapshot"
python3 - "$project_root" "$snapshot_dir" <<'PY'
import sys
from pathlib import Path
root, snapshot = (Path(value).resolve() for value in sys.argv[1:])
if snapshot.parent != root / ".repro/snapshots" or not snapshot.name.startswith("snapshot-"):
    raise SystemExit("O snapshot deve pertencer ao diretório privado deste ensaio.")
PY

repro_kubectl=(kubectl --kubeconfig "$project_root/.repro/kubeconfig" --context kind-energy-mlops-repro --namespace energy-mlops-repro)
target_nodes="$("${repro_kubectl[@]}" get nodes -o 'jsonpath={.items[*].metadata.name}')"
[[ "$target_nodes" == energy-mlops-repro-control-plane ]] || { echo "Node do ensaio não confirmado." >&2; exit 1; }
api_deployment="$("${repro_kubectl[@]}" get deployment energy-api --ignore-not-found -o name)"
[[ -z "$api_deployment" ]] || { echo "A API deve permanecer desabilitada durante a restauração." >&2; exit 1; }

echo "Revalidando o snapshot contra os hashes registrados..."
python3 "$project_root/scripts/snapshot-repro-objects.py" manifest --directory "$snapshot_dir" \
  > "$project_root/.repro/restore-manifest.json"
bash "$project_root/scripts/inventory-repro-data.sh" > "$project_root/.repro/restore-preflight.log"

# Check tracking entities too: an empty Registry alone does not prove an empty DB.
empty_sql='SELECT (SELECT count(*) FROM runs) + (SELECT count(*) FROM registered_models) + (SELECT count(*) FROM model_versions) + (SELECT count(*) FROM logged_models) + (SELECT count(*) FROM experiments WHERE experiment_id <> 0);'
db_rows="$("${repro_kubectl[@]}" exec deployment/postgres -- sh -c \
  'export PGPASSWORD="$POSTGRES_PASSWORD"; exec psql --no-psqlrc --no-password --tuples-only --no-align --set=ON_ERROR_STOP=1 --host=127.0.0.1 --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --command="$1"' sh "$empty_sql")"
[[ "$db_rows" == 0 ]] || { echo "O banco de destino contém dados de aplicação; restauração recusada." >&2; exit 1; }

object_script="$(< "$project_root/scripts/snapshot-repro-objects.py")"
echo "Restaurando os objetos no RustFS do ensaio..."
restore_phase=objects
{ cat "$project_root/.repro/restore-manifest.json"; cat "$snapshot_dir/objects.tar"; } \
  | "${repro_kubectl[@]}" exec -i deployment/mlflow -- python3 -c "$object_script" restore \
  > "$project_root/.repro/restored-objects.json"

# Temporary suspension is confined to the rehearsal and returns to one replica on success.
echo "Suspendendo somente o MLflow do ensaio para restaurar o banco..."
restore_phase=database
"${repro_kubectl[@]}" scale deployment/mlflow --replicas=0
# Poll the empty pod list, including the case where deletion finished immediately.
for ((attempt=0; attempt<45; attempt++)); do
  remaining_pods="$("${repro_kubectl[@]}" get pods -l app=mlflow -o name)"
  [[ -z "$remaining_pods" ]] && break
  sleep 2
done
[[ -z "$remaining_pods" ]] || { echo "Pods MLflow do ensaio não encerraram em 90s." >&2; exit 1; }
"${repro_kubectl[@]}" exec -i deployment/postgres -- sh -c \
  'export PGPASSWORD="$POSTGRES_PASSWORD"; exec pg_restore --no-password --clean --if-exists --single-transaction --exit-on-error --no-owner --no-acl --host=127.0.0.1 --username="$POSTGRES_USER" --dbname="$POSTGRES_DB"' \
  < "$snapshot_dir/postgres.dump"
"${repro_kubectl[@]}" scale deployment/mlflow --replicas=1
"${repro_kubectl[@]}" rollout status deployment/mlflow --timeout=180s

echo "Conferindo Registry e hashes dos objetos restaurados..."
restore_phase=validation
"${repro_kubectl[@]}" exec -i deployment/mlflow -- python3 -c "$object_script" check \
  < "$project_root/.repro/restore-manifest.json" > "$project_root/.repro/restored-data.json"
curl -fsS --max-time 10 http://localhost:8000/model-info > "$project_root/.repro/source-after-restore.json"
python3 - "$project_root/.repro/source-after-restore.json" <<'PY'
import json, sys
with open(sys.argv[1]) as source:
    model = json.load(source)
if model["model_name"] != "ensemble_lgb_xgb_rf_bahia" or model["version"] != 17 or model["run_id"] != "1d13a61244c54f06aa70f43a9993ea37":
    raise SystemExit("A API de origem não confirmou a identidade v17 esperada.")
PY
cat "$project_root/.repro/restored-data.json"
"${repro_kubectl[@]}" get deployments,pods,pvc
echo "Restauração verificada no ensaio. A API de destino continua desabilitada."
