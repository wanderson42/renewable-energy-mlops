#!/usr/bin/env bash
set -euo pipefail
umask 077

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for tool in kubectl python3 curl; do
  command -v "$tool" >/dev/null || { echo "Ferramenta ausente: $tool" >&2; exit 1; }
done
[[ -f "$project_root/.repro/kubeconfig" ]] || { echo "Kubeconfig de ensaio ausente." >&2; exit 1; }
mkdir -p "$project_root/.repro/snapshots"
snapshot_dir="$(mktemp -d "$project_root/.repro/snapshots/snapshot-$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")"
trap 'echo "Backup interrompido. Arquivos parciais preservados em: $snapshot_dir" >&2' ERR
touch "$snapshot_dir/INCOMPLETE"
echo "Snapshot: $snapshot_dir"
echo "Execute durante um intervalo sem gravações de treino, promoção, ingestão ou monitoring."

bash "$project_root/scripts/inventory-repro-data.sh" > "$snapshot_dir/preflight.log"
cp -- "$project_root/.repro/source-data-inventory.json" "$snapshot_dir/source-before.json"
cp -- "$project_root/.repro/destination-data-inventory.json" "$snapshot_dir/destination-before.json"

kubectl --context kind-energy-mlops --namespace default \
  exec -i deployment/mlflow -- python3 - inventory \
  < "$project_root/scripts/snapshot-repro-objects.py" > "$snapshot_dir/objects-before.json"

python3 - "$snapshot_dir" <<'PY'
import json, shutil, sys
from pathlib import Path
directory = Path(sys.argv[1])
inventory = json.loads((directory / "objects-before.json").read_text())
required = sum(item["size"] for item in inventory["objects"]) + 2 * 1024**3
if shutil.disk_usage(directory).free < required:
    raise SystemExit("Espaço livre insuficiente para objetos e reserva de 2 GiB para o banco.")
PY

echo "Capturando PostgreSQL..."
kubectl --context kind-energy-mlops --namespace default exec deployment/postgres -- \
  sh -c 'export PGPASSWORD="$POSTGRES_PASSWORD"; exec pg_dump --no-password --format=custom --no-acl --host=127.0.0.1 --username="$POSTGRES_USER" --dbname="$POSTGRES_DB"' \
  > "$snapshot_dir/postgres.dump"

echo "Capturando os objetos S3..."
object_script="$(< "$project_root/scripts/snapshot-repro-objects.py")"
kubectl --context kind-energy-mlops --namespace default \
  exec -i deployment/mlflow -- python3 -c "$object_script" export \
  < "$snapshot_dir/objects-before.json" > "$snapshot_dir/objects.tar"

# Generate SQL into /dev/null to parse/decompress the dump without connecting to a DB.
kubectl --context kind-energy-mlops --namespace default exec -i deployment/postgres -- \
  pg_restore --no-owner --no-acl --file=/dev/null < "$snapshot_dir/postgres.dump"

curl -fsS --max-time 10 http://localhost:8000/model-info > "$snapshot_dir/source-model-info.json"
python3 - "$snapshot_dir/source-model-info.json" <<'PY'
import json, sys
with open(sys.argv[1]) as source:
    model = json.load(source)
if model["model_name"] != "ensemble_lgb_xgb_rf_bahia" or model["version"] != 17 or model["run_id"] != "1d13a61244c54f06aa70f43a9993ea37":
    raise SystemExit("A API de origem não confirmou a identidade v17 esperada.")
PY

echo "Verificando hashes e estrutura do snapshot..."
python3 "$project_root/scripts/snapshot-repro-objects.py" verify --directory "$snapshot_dir"
rm -- "$snapshot_dir/INCOMPLETE"
printf '%s\n' "$snapshot_dir" > "$project_root/.repro/latest-snapshot"
echo "Backup verificado. Nenhum dado foi restaurado no destino."
