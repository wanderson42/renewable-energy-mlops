# Testes de integração da FastAPI:
# src/energy_mlops/service/main.py

import pytest
from fastapi.testclient import TestClient

from energy_mlops.service import main as service_main

app = service_main.app
model_cache = service_main.model_cache
model_metadata_cache = service_main.model_metadata_cache


# ==========================================
# 1. MODELO FALSO
# ==========================================
class FakeModel:
    """
    Simula um modelo MLflow que sempre prevê
    50% de Fator de Capacidade.
    """

    def predict(self, df):
        return [0.5] * len(df)


# ==========================================
# 2. FIXTURES
# ==========================================
@pytest.fixture
def fake_champion_metadata():
    """
    Metadados falsos equivalentes aos retornados
    por load_champion_metadata().
    """

    return {
        "model_name": "ensemble_lgb_xgb_rf_bahia",
        "alias": "champion",
        "version": 10,
        "run_id": "fake-run-id-123",
        "metrics": {
            "oot_mae_mw": 823.5,
            "oot_nmae_pct": 7.04,
            "oot_mae_fc_pct": 6.91,
            "oot_r2_score": 0.82,
        },
    }


@pytest.fixture(autouse=True)
def clear_model_caches():
    """
    Garante isolamento entre os testes.

    Nenhum teste deve herdar modelo ou metadados
    deixados por outro teste.
    """

    model_cache.clear()
    model_metadata_cache.clear()

    yield

    model_cache.clear()
    model_metadata_cache.clear()


@pytest.fixture
def mock_mlflow_success(
    mocker,
    fake_champion_metadata,
):
    """
    Simula startup saudável sem acessar
    MLflow ou RustFS reais.
    """

    mock_load_model = mocker.patch.object(
        service_main,
        "load_champion_model",
        return_value=FakeModel(),
    )

    mock_load_metadata = mocker.patch.object(
        service_main,
        "load_champion_metadata",
        return_value=fake_champion_metadata,
    )

    return (
        mock_load_model,
        mock_load_metadata,
    )


@pytest.fixture
def mock_mlflow_failure(
    mocker,
    fake_champion_metadata,
):
    """
    Simula falha durante o carregamento
    do Champion.
    """

    mock_load_model = mocker.patch.object(
        service_main,
        "load_champion_model",
        side_effect=Exception(
            "MLflow Server Down!"
        ),
    )

    mock_load_metadata = mocker.patch.object(
        service_main,
        "load_champion_metadata",
        return_value=fake_champion_metadata,
    )

    return (
        mock_load_model,
        mock_load_metadata,
    )


@pytest.fixture
def client_healthy(
    mock_mlflow_success,
):
    """
    O context manager aciona o lifespan da API.

    No startup:
    - carrega FakeModel;
    - carrega metadados falsos do Champion.
    """

    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_unhealthy(
    mock_mlflow_failure,
):
    """
    Cliente cuja inicialização do modelo
    falha durante o lifespan.
    """

    with TestClient(app) as client:
        yield client


# ==========================================
# 3. HEALTH CHECK
# ==========================================
def test_health_check_healthy(
    client_healthy,
):
    """
    Kubernetes deve receber readiness 200
    quando o Champion estiver carregado.
    """

    response = client_healthy.get(
        "/health"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_health_check_unhealthy(
    client_unhealthy,
):
    """
    Se o Champion não puder ser carregado,
    /health deve retornar 503.
    """

    response = client_unhealthy.get(
        "/health"
    )

    assert response.status_code == 503

    body = response.json()

    assert (
        body["detail"]["status"]
        == "unhealthy"
    )


# ==========================================
# 4. MODEL INFO
# ==========================================
def test_model_info_returns_champion_metadata(
    client_healthy,
    fake_champion_metadata,
):
    """
    /model-info deve representar exatamente
    o Champion carregado pela API.
    """

    response = client_healthy.get(
        "/model-info"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["model_name"] == (
        "ensemble_lgb_xgb_rf_bahia"
    )

    assert body["alias"] == "champion"
    assert body["version"] == 10
    assert body["run_id"] == "fake-run-id-123"

    assert body["metrics"] == {
        "oot_mae_mw": 823.5,
        "oot_nmae_pct": 7.04,
        "oot_mae_fc_pct": 6.91,
        "oot_r2_score": 0.82,
    }

    assert body == fake_champion_metadata


def test_model_info_returns_503_when_metadata_unavailable(
    client_healthy,
):
    """
    Mesmo com modelo carregado, /model-info
    não deve fabricar metadados caso eles
    estejam indisponíveis.
    """

    assert (
        model_cache.get("champion")
        is not None
    )

    model_metadata_cache.clear()

    response = client_healthy.get(
        "/model-info"
    )

    assert response.status_code == 503

    body = response.json()

    assert (
        "Champion"
        in str(body["detail"])
    )


# ==========================================
# 5. INFERÊNCIA
# ==========================================
def test_predict_batch_success(
    client_healthy,
):
    """
    Fluxo:

    request
        ->
    feature engineering
        ->
    modelo
        ->
    predicted_fc
        ->
    predicted_mw
    """

    payload = {
        "predictions": [
            {
                "date": (
                    "2026-09-24T12:00:00Z"
                ),
                "wind_speed_100m": 12.5,
                "wind_direction_100m": 180.0,
                "temperature_2m": 25.0,
            }
        ]
    }

    response = client_healthy.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1

    # FakeModel sempre retorna FC = 0.5
    assert (
        data[0]["predicted_fc"]
        == pytest.approx(0.5)
    )

    # capacidade_mw foi adicionada pelo
    # pipeline de feature engineering.
    assert (
        data[0]["predicted_mw"]
        is not None
    )

    assert (
        data[0]["predicted_mw"]
        > 0
    )


def test_predict_batch_invalid_payload(
    client_healthy,
):
    """
    Pydantic deve rejeitar vento negativo
    antes de chegar ao modelo.
    """

    payload = {
        "predictions": [
            {
                "date": (
                    "2026-09-24T12:00:00Z"
                ),
                "wind_speed_100m": -10.0,
                "wind_direction_100m": 180.0,
                "temperature_2m": 25.0,
            }
        ]
    }

    response = client_healthy.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422


# ==========================================
# 6. HOT RELOAD
# ==========================================
def test_reload_model_updates_model_and_metadata(
    client_healthy,
    mock_mlflow_success,
):
    """
    Hot reload deve atualizar conjuntamente:

    - modelo servido;
    - versão;
    - run_id;
    - métricas do Champion.
    """

    (
        mock_load_model,
        mock_load_metadata,
    ) = mock_mlflow_success

    new_model = FakeModel()

    new_metadata = {
        "model_name": (
            "ensemble_lgb_xgb_rf_bahia"
        ),
        "alias": "champion",
        "version": 11,
        "run_id": "new-run-id-456",
        "metrics": {
            "oot_mae_mw": 750.0,
            "oot_nmae_pct": 6.42,
            "oot_mae_fc_pct": 6.20,
            "oot_r2_score": 0.86,
        },
    }

    # A partir daqui simulamos que o alias
    # Champion passou da v10 para v11.
    mock_load_model.return_value = (
        new_model
    )

    mock_load_metadata.return_value = (
        new_metadata
    )

    response = client_healthy.post(
        "/reload-model"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "success"
    assert body["version"] == 11

    assert (
        model_cache["champion"]
        is new_model
    )

    assert (
        model_metadata_cache["champion"]
        == new_metadata
    )

    # /model-info também deve refletir
    # imediatamente o novo Champion.
    info_response = client_healthy.get(
        "/model-info"
    )

    assert (
        info_response.status_code
        == 200
    )

    assert (
        info_response.json()["version"]
        == 11
    )

    assert (
        info_response.json()["run_id"]
        == "new-run-id-456"
    )


def test_reload_model_does_not_leave_partial_state(
    client_healthy,
    mock_mlflow_success,
):
    """
    Se o novo modelo for baixado, mas houver
    falha ao obter seus metadados, a API não
    deve ficar com:

        modelo = v11
        metadata = v10

    O estado anterior deve ser preservado.
    """

    (
        mock_load_model,
        mock_load_metadata,
    ) = mock_mlflow_success

    old_model = model_cache[
        "champion"
    ]

    old_metadata = (
        model_metadata_cache[
            "champion"
        ].copy()
    )

    # Download do novo modelo funcionou...
    mock_load_model.return_value = (
        FakeModel()
    )

    # ...mas a obtenção dos metadados falhou.
    mock_load_metadata.side_effect = (
        Exception(
            "Metadata unavailable!"
        )
    )

    response = client_healthy.post(
        "/reload-model"
    )

    assert response.status_code == 500

    # Estado anterior deve permanecer intacto.
    assert (
        model_cache["champion"]
        is old_model
    )

    assert (
        model_metadata_cache["champion"]
        == old_metadata
    )

def test_resolve_metric_uses_canonical_metric():
    metrics = {
        "oot_nmae_pct": 6.9,
    }

    assert (
        service_main.resolve_metric(
            metrics,
            "oot_nmae_pct",
        )
        == 6.9
    )


def test_resolve_metric_accepts_legacy_year():
    metrics = {
        "oot_nmae_pct_2026": 7.04,
    }

    assert (
        service_main.resolve_metric(
            metrics,
            "oot_nmae_pct",
        )
        == 7.04
    )


def test_resolve_metric_returns_none_when_absent():
    assert (
        service_main.resolve_metric(
            {},
            "oot_nmae_pct",
        )
        is None
    )