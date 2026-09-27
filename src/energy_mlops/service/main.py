import os
from contextlib import asynccontextmanager

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from energy_mlops.config import settings
from energy_mlops.data.build_features import generate_wind_and_time_features
from energy_mlops.service.schema import BatchPredictionRequest, PredictionResponse

# Mapeia variáveis de ambiente priorizando o ambiente do sistema (Kubernetes) ou o settings local
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("RUSTFS_ROOT_USER", settings.RUSTFS_ROOT_USER)
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("RUSTFS_ROOT_PASSWORD", settings.RUSTFS_ROOT_PASSWORD)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("RUSTFS_ENDPOINT", settings.RUSTFS_ENDPOINT)

model_cache = {}
MODEL_NAME = "ensemble_lgb_xgb_rf_bahia"  # Nome registrado no MLflow em train_ensemble.py


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida do FastAPI: Baixa o modelo @champion no startup."""
    model_uri = f"models:/{MODEL_NAME}@champion"
    print(f"🔄 Conectando ao MLflow e baixando {model_uri}...")

    try:
        model_cache["champion"] = mlflow.pyfunc.load_model(model_uri)
        print("✅ Modelo @champion carregado na memória com sucesso!")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Erro ao carregar o modelo no startup: {e}")
        model_cache["champion"] = None

    yield

    model_cache.clear()


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


@app.post("/reload-model", status_code=200)
async def reload_model():
    """Endpoint de Hot-Reload: Recarrega o modelo @champion mais recente do MLflow para a RAM."""
    model_uri = f"models:/{MODEL_NAME}@champion"
    print(f"🔄 Solicitação de Hot-Reload recebida. Baixando {model_uri}...")
    try:
        model_cache["champion"] = mlflow.pyfunc.load_model(model_uri)
        print("✅ Modelo @champion atualizado com sucesso na memória!")
        return {"status": "success", "message": "Modelo recarregado com sucesso na memória."}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha ao recarregar o modelo: {e!s}",
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