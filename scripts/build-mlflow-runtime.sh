#!/usr/bin/env bash
set -euo pipefail
umask 077

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v docker >/dev/null || { echo "Docker ausente." >&2; exit 1; }
cd "$project_root"
git_revision="$(git rev-parse HEAD)"
runtime_image="energy-mlops-mlflow:${git_revision}"
mkdir -p .repro

docker build --platform linux/amd64 \
  --label "org.opencontainers.image.revision=$git_revision" \
  --label "org.opencontainers.image.source=https://github.com/wanderson42/renewable-energy-mlops" \
  --tag "$runtime_image" docker/mlflow
bash scripts/test-mlflow-runtime.sh "$runtime_image"
docker image inspect "$runtime_image" \
  --format '{"local_image_id":{{json .Id}},"local_tag":{{json (index .RepoTags 0)}}}' \
  > .repro/mlflow-runtime-build.json
cat .repro/mlflow-runtime-build.json
echo "Imagem local validada. Nenhuma imagem publicada e nenhum workload atualizado."
