import os
import re
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from mlflow.tracking import MlflowClient

from energy_mlops.config import settings
from energy_mlops.data.build_features import (
    generate_wind_and_time_features,
)
from energy_mlops.service.schema import (
    BatchPredictionRequest,
    PredictionResponse,
)

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

MODEL_NAME = "ensemble_lgb_xgb_rf_bahia"
MODEL_ALIAS = "champion"


# Preserva variáveis explicitamente fornecidas pelo ambiente
# (Kubernetes, shell, CI etc.) e usa settings apenas como fallback.
os.environ["MLFLOW_TRACKING_URI"] = os.getenv(
    "MLFLOW_TRACKING_URI",
    settings.MLFLOW_TRACKING_URI,
)

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv(
    "AWS_ACCESS_KEY_ID",
    os.getenv(
        "RUSTFS_ROOT_USER",
        settings.RUSTFS_ROOT_USER,
    ),
)

os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv(
    "AWS_SECRET_ACCESS_KEY",
    os.getenv(
        "RUSTFS_ROOT_PASSWORD",
        settings.RUSTFS_ROOT_PASSWORD,
    ),
)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
    "MLFLOW_S3_ENDPOINT_URL",
    os.getenv(
        "RUSTFS_ENDPOINT",
        settings.RUSTFS_ENDPOINT,
    ),
)


# ==============================================================================
# CACHE OPERACIONAL
# ==============================================================================

model_cache = {}
model_metadata_cache = {}


# ==============================================================================
# COMPATIBILIDADE DE MÉTRICAS
# ==============================================================================

def resolve_metric(
    metrics: Mapping[str, Any],
    canonical_name: str,
) -> float | None:
    """
    Resolve uma métrica para o contrato operacional da API.

    A API sempre expõe o nome canônico.

    Se uma Run histórica não possuir a chave canônica,
    é aceita uma única variante com sufixo anual:

        <canonical_name>_YYYY

    Nenhum ano específico é codificado na aplicação.

    Caso existam múltiplas variantes anuais e nenhuma
    chave canônica, a função não escolhe silenciosamente
    uma delas, pois a origem seria ambígua.
    """

    canonical_value = metrics.get(
        canonical_name
    )

    if canonical_value is not None:
        return float(canonical_value)

    legacy_pattern = re.compile(
        rf"^{re.escape(canonical_name)}_(\d{{4}})$"
    )

    legacy_candidates = []

    for metric_name, metric_value in metrics.items():
        if metric_value is None:
            continue

        if legacy_pattern.fullmatch(
            metric_name
        ):
            legacy_candidates.append(
                (
                    metric_name,
                    metric_value,
                )
            )

    if len(legacy_candidates) == 1:
        _, metric_value = (
            legacy_candidates[0]
        )

        return float(metric_value)

    if len(legacy_candidates) > 1:
        candidates = ", ".join(
            name
            for name, _ in legacy_candidates
        )

        raise RuntimeError(
            f"Múltiplas variantes históricas "
            f"encontradas para '{canonical_name}': "
            f"{candidates}."
        )

    return None


# ==============================================================================
# ACESSO AO CHAMPION
# ==============================================================================

def load_champion_model():
    """
    Carrega o modelo registrado com o alias Champion.

    A função cria uma fronteira explícita entre FastAPI
    e MLflow, permitindo isolamento adequado nos testes.
    """

    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    return mlflow.pyfunc.load_model(
        model_uri
    )


def load_champion_metadata() -> dict:
    """
    Obtém os metadados operacionais da mesma versão
    do modelo identificada pelo alias Champion.

    Independentemente do schema histórico da Run,
    o contrato retornado pela API usa somente nomes
    canônicos de métricas.
    """

    client = MlflowClient(
        tracking_uri=(
            settings.MLFLOW_TRACKING_URI
        )
    )

    model_version = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS,
        )
    )

    run = client.get_run(
        model_version.run_id
    )

    metrics = run.data.metrics

    return {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": int(
            model_version.version
        ),
        "run_id": model_version.run_id,
        "metrics": {
            "oot_mae_mw": resolve_metric(
                metrics,
                "oot_mae_mw",
            ),
            "oot_nmae_pct": resolve_metric(
                metrics,
                "oot_nmae_pct",
            ),
            "oot_mae_fc_pct": resolve_metric(
                metrics,
                "oot_mae_fc_pct",
            ),
            "oot_r2_score": resolve_metric(
                metrics,
                "oot_r2_score",
            ),
        },
    }


# ==============================================================================
# LIFESPAN
# ==============================================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    print(
        "🔄 Conectando ao MLflow e baixando "
        f"{model_uri}..."
    )

    try:
        new_model = (
            load_champion_model()
        )

        new_metadata = (
            load_champion_metadata()
        )

        # Commit conjunto do estado operacional somente
        # após modelo e metadados terem sido carregados.
        model_cache[
            "champion"
        ] = new_model

        model_metadata_cache[
            "champion"
        ] = new_metadata

        print(
            "✅ Modelo @champion e metadados "
            "carregados com sucesso!"
        )

    except Exception as exc:  # noqa: BLE001
        print(
            "⚠️ Erro ao carregar Champion: "
            f"{exc}"
        )

        model_cache[
            "champion"
        ] = None

        model_metadata_cache[
            "champion"
        ] = None

    yield

    model_cache.clear()
    model_metadata_cache.clear()


# ==============================================================================
# FASTAPI
# ==============================================================================

app = FastAPI(
    title="Wind Energy Forecast API",
    description=(
        "API de Inferência em Tempo Real "
        "para Geração Eólica na Bahia"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ==============================================================================
# HEALTH
# ==============================================================================

@app.get("/health")
async def health_check():
    """
    Readiness/Liveness operacional da API.
    """

    is_ready = (
        model_cache.get(
            "champion"
        )
        is not None
    )

    if not is_ready:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail={
                "status": "unhealthy",
                "model_loaded": False,
            },
        )

    return {
        "status": "healthy",
        "model_loaded": True,
    }


# ==============================================================================
# MODEL INFO
# ==============================================================================

@app.get("/model-info")
async def model_info():
    """
    Retorna a verdade operacional do Champion
    atualmente carregado pela aplicação.
    """

    metadata = (
        model_metadata_cache.get(
            "champion"
        )
    )

    if metadata is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Metadados do modelo Champion "
                "não estão disponíveis."
            ),
        )

    return metadata


# ==============================================================================
# HOT RELOAD
# ==============================================================================

@app.post(
    "/reload-model",
    status_code=200,
)
async def reload_model():
    """
    Recarrega o Champion mais recente do MLflow.

    Modelo e metadados são primeiro carregados em
    variáveis temporárias. O cache operacional só é
    atualizado depois que ambos forem obtidos com
    sucesso, evitando estado parcialmente atualizado.
    """

    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    print(
        "🔄 Solicitação de Hot-Reload recebida. "
        f"Baixando {model_uri}..."
    )

    try:
        new_model = (
            load_champion_model()
        )

        new_metadata = (
            load_champion_metadata()
        )

        # Atualização conjunta após sucesso dos dois loads.
        model_cache[
            "champion"
        ] = new_model

        model_metadata_cache[
            "champion"
        ] = new_metadata

        return {
            "status": "success",
            "message": (
                "Modelo e metadados Champion "
                "recarregados com sucesso."
            ),
            "version": (
                new_metadata[
                    "version"
                ]
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Falha ao recarregar o modelo: "
                f"{exc!s}"
            ),
        ) from exc


# ==============================================================================
# BATCH INFERENCE
# ==============================================================================

@app.post(
    "/predict/batch",
    response_model=list[
        PredictionResponse
    ],
)
def predict_batch(
    payload: BatchPredictionRequest,
):
    """
    Executa inferência em lote utilizando
    o Champion carregado em memória.
    """

    model = model_cache.get(
        "champion"
    )

    if model is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "O modelo preditivo não está "
                "disponível no servidor."
            ),
        )

    try:
        # --------------------------------------------------------------
        # 1. Payload -> DataFrame
        # --------------------------------------------------------------

        raw_dicts = [
            item.model_dump()
            for item
            in payload.predictions
        ]

        df_raw = pd.DataFrame(
            raw_dicts
        )

        # --------------------------------------------------------------
        # 2. Engenharia de features
        # --------------------------------------------------------------

        df_features = (
            generate_wind_and_time_features(
                df_raw
            )
        )

        # --------------------------------------------------------------
        # 3. Predição do fator de capacidade
        # --------------------------------------------------------------

        preds_fc = model.predict(
            df_features
        )

        preds_fc = np.clip(
            np.asarray(
                preds_fc
            ),
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # 4. Conversão FC -> MW
        # --------------------------------------------------------------

        preds_mw = (
            preds_fc
            * df_features[
                "capacidade_mw"
            ].to_numpy()
        )

        # --------------------------------------------------------------
        # 5. Contrato HTTP de resposta
        # --------------------------------------------------------------

        response = []

        for item, fc, mw in zip(
            payload.predictions,
            preds_fc,
            preds_mw,
            strict=True,
        ):
            response.append(
                PredictionResponse(
                    date=item.date,
                    predicted_fc=float(fc),
                    predicted_mw=float(mw),
                )
            )

        return response

    except Exception as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Erro durante o processamento "
                f"da inferência: {exc!s}"
            ),
        ) from exc