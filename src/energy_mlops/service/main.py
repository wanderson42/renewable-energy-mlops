import os
from contextlib import asynccontextmanager

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from mlflow.tracking import MlflowClient

from energy_mlops.config import settings
from energy_mlops.data.build_features import generate_wind_and_time_features
from energy_mlops.service.schema import BatchPredictionRequest, PredictionResponse

# Mapeia variáveis de ambiente priorizando o ambiente do sistema (Kubernetes) ou o settings local
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("RUSTFS_ROOT_USER", settings.RUSTFS_ROOT_USER)
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("RUSTFS_ROOT_PASSWORD", settings.RUSTFS_ROOT_PASSWORD)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("RUSTFS_ENDPOINT", settings.RUSTFS_ENDPOINT)

model_cache = {}
model_metadata_cache = {}

MODEL_NAME = "ensemble_lgb_xgb_rf_bahia"
MODEL_ALIAS = "champion"

def load_champion_model():
    """
    Carrega o modelo registrado com o alias Champion.

    Esta função centraliza o acesso ao MLflow
    e cria uma fronteira explícita para testes.
    """
    model_uri = (f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

    return mlflow.pyfunc.load_model(model_uri)

def load_champion_metadata() -> dict:
    client = MlflowClient(
        tracking_uri=settings.MLFLOW_TRACKING_URI
    )

    model_version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )

    run = client.get_run(
        model_version.run_id
    )

    metrics = run.data.metrics

    return {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": int(model_version.version),
        "run_id": model_version.run_id,
        "metrics": {
            "oot_mae_mw": metrics.get(
                "oot_mae_mw"
            ),
            "oot_nmae_pct": metrics.get(
                "oot_nmae_pct"
            ),
            "oot_mae_fc_pct": metrics.get(
                "oot_mae_fc_pct"
            ),
            "oot_r2_score": metrics.get(
                "oot_r2_score"
            ),
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    print(
        f"🔄 Conectando ao MLflow e baixando "
        f"{model_uri}..."
    )

    try:

        model_cache["champion"] = (
            load_champion_model()
        )

        model_metadata_cache["champion"] = (
            load_champion_metadata()
        )

        print(
            "✅ Modelo @champion e metadados "
            "carregados com sucesso!"
        )

    except Exception as e:  # noqa: BLE001
        print(
            f"⚠️ Erro ao carregar Champion: {e}"
        )

        model_cache["champion"] = None
        model_metadata_cache["champion"] = None

    yield

    model_cache.clear()
    model_metadata_cache.clear()


app = FastAPI(
    title="Wind Energy Forecast API",
    description="API de Inferência em Tempo Real para Geração Eólica na Bahia",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Endpoint de Readiness/Liveness probe para o Kubernetes."""
    is_ready = model_cache.get("champion") is not None
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "unhealthy", "model_loaded": False},
        )
    return {"status": "healthy", "model_loaded": True}

@app.get("/model-info")
async def model_info():
    metadata = model_metadata_cache.get(
        "champion"
    )

    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Metadados do modelo Champion "
                "não estão disponíveis."
            ),
        )

    return metadata

@app.post("/reload-model", status_code=200)
async def reload_model():
    """Endpoint de Hot-Reload: Recarrega o modelo @champion mais recente do
      MLflow para a RAM e atualiza os metadados."""
    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    print(
        "🔄 Solicitação de Hot-Reload recebida. "
        f"Baixando {model_uri}..."
    )

    try:

        new_model = load_champion_model()

        new_metadata = (
            load_champion_metadata()
        )

        model_cache["champion"] = new_model

        model_metadata_cache[
            "champion"
        ] = new_metadata

        return {
            "status": "success",
            "message": (
                "Modelo e metadados Champion "
                "recarregados com sucesso."
            ),
            "version": new_metadata["version"],
        }

    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                f"Falha ao recarregar o modelo: {e!s}"
            ),
        )

@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(payload: BatchPredictionRequest):
    model = model_cache.get("champion")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O modelo preditivo não está disponível no servidor.",
        )

    try:
        # 1. Converter payload para DataFrame
        raw_dicts = [item.model_dump() for item in payload.predictions]
        df_raw = pd.DataFrame(raw_dicts)

        # 2. Aplicar pipeline de features (garante 'capacidade_mw' preenchida via merge_asof)
        df_features = generate_wind_and_time_features(df_raw)

        # 3. Predição do Fator de Capacidade
        preds_fc = model.predict(df_features)
        preds_fc = np.clip(np.array(preds_fc), 0.0, 1.0)

        # 4. Cálculo direto usando a coluna fornecida pelo pipeline
        preds_mw = preds_fc * df_features["capacidade_mw"].values

        # 5. Montagem da resposta
        response = []
        for item, fc, mw in zip(payload.predictions, preds_fc, preds_mw):
            response.append(
                PredictionResponse(
                    date=item.date,
                    predicted_fc=float(fc),
                    predicted_mw=float(mw),
                )
            )

        return response

    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro durante o processamento da inferência: {str(e)}",  # noqa: RUF010
        )