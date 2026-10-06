#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_NAME="energy-mlops-repro"
CLUSTER_CONTEXT="kind-${CLUSTER_NAME}"
KUBECONFIG_PATH="${PROJECT_ROOT}/.repro/kubeconfig"

for tool in docker kind kubectl; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Pré-requisito ausente: $tool" >&2
        exit 1
    fi
done

# Usa o mesmo runtime do ambiente de origem, sem depender de autodetecção.
export KIND_EXPERIMENTAL_PROVIDER=docker
docker info >/dev/null
clusters="$(kind get clusters)"
while IFS= read -r cluster; do
    if [[ "$cluster" == "$CLUSTER_NAME" ]]; then
        echo "O cluster ${CLUSTER_NAME} já existe; não será recriado." >&2
        exit 1
    fi
done <<< "$clusters"

if [[ -e "$KUBECONFIG_PATH" || -L "$KUBECONFIG_PATH" ]]; then
    echo "O kubeconfig de ensaio já existe; não será sobrescrito." >&2
    exit 1
fi

umask 077
mkdir -p "${PROJECT_ROOT}/.repro"
chmod 700 "${PROJECT_ROOT}/.repro"

kind create cluster \
    --name "$CLUSTER_NAME" \
    --config "${PROJECT_ROOT}/infra/kind/repro.yaml" \
    --kubeconfig "$KUBECONFIG_PATH" \
    --wait 120s

chmod 600 "$KUBECONFIG_PATH"
kubectl --kubeconfig "$KUBECONFIG_PATH" --context "$CLUSTER_CONTEXT" \
    wait --for=condition=Ready nodes --all --timeout=120s
kubectl --kubeconfig "$KUBECONFIG_PATH" --context "$CLUSTER_CONTEXT" \
    get nodes -o wide

echo "Cluster de ensaio criado. Kubeconfig: ${KUBECONFIG_PATH}"
echo "Nenhuma aplicação, release Helm ou modelo foi instalado."
