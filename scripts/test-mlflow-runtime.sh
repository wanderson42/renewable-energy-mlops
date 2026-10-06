#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 && -n "$1" ]] || { echo "Use: bash scripts/test-mlflow-runtime.sh IMAGE" >&2; exit 1; }
command -v docker >/dev/null || { echo "Docker ausente." >&2; exit 1; }
runtime_image="$1"
smoke_container=""
cleanup() {
  if [[ -n "$smoke_container" ]]; then
    docker rm -f "$smoke_container" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# No published ports, shared volumes, credentials or host network.
docker run --rm --network none "$runtime_image" \
  python3 /opt/energy-mlops-mlflow/verify_runtime.py
smoke_container="$(docker run -d --network none --memory 2g --cpus 2 \
  --tmpfs /tmp:rw,size=256m "$runtime_image" \
  mlflow server --backend-store-uri sqlite:////tmp/mlflow-smoke.db \
  --default-artifact-root /tmp/mlflow-smoke-artifacts \
  --host 127.0.0.1 --port 5000 --workers 1 --allowed-hosts '*')"
if ! docker exec "$smoke_container" python3 /opt/energy-mlops-mlflow/verify_runtime.py \
  --server-url http://127.0.0.1:5000; then
  docker logs --tail 80 "$smoke_container" >&2 || true
  exit 1
fi
echo "Runtime e HTTP do MLflow aprovados sem rede externa."
