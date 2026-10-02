#!/usr/bin/env bash

set -u

PF_PID=""

cleanup_child() {
    if [[ -n "${PF_PID}" ]] && kill -0 "${PF_PID}" 2>/dev/null; then
        kill "${PF_PID}" 2>/dev/null || true
        wait "${PF_PID}" 2>/dev/null || true
    fi

    PF_PID=""
}

terminate() {
    cleanup_child
    exit 0
}

trap terminate INT TERM
trap cleanup_child EXIT

while true; do
    echo "[$(date -Is)] Iniciando MLflow port-forward..."

    kubectl port-forward svc/mlflow 5000:5000 &
    PF_PID=$!

    wait "${PF_PID}"
    EXIT_CODE=$?
    PF_PID=""

    echo \
        "[$(date -Is)] MLflow port-forward encerrou " \
        "(exit=${EXIT_CODE}). Reiniciando em 2s..."

    sleep 2
done