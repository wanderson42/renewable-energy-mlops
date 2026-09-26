# Testes de integração da FastAPI (rotas HTTP): src/energy_mlops/service/main.py
import pytest
from fastapi.testclient import TestClient

# Ajuste o import conforme a árvore do seu projeto
from energy_mlops.service.main import app


# ==========================================
# 1. Criação do Modelo Falso (Mock)
# ==========================================
class FakeModel:
    """Simula um modelo do MLflow que sempre prevê 50% de Fator de Capacidade."""
    def predict(self, df):
        # Retorna uma lista de 0.5 (um para cada linha do DataFrame enviado)
        return [0.5] * len(df)

# ==========================================
# 2. Fixtures (Preparação de Ambiente)
# ==========================================
@pytest.fixture
def mock_mlflow_success(mocker):
    """Intercepta a chamada ao MLflow e injeta nosso FakeModel."""
    return mocker.patch("mlflow.pyfunc.load_model", return_value=FakeModel())

@pytest.fixture
def mock_mlflow_failure(mocker):
    """Simula uma queda do servidor do MLflow (Timeout ou Erro)."""
    return mocker.patch("mlflow.pyfunc.load_model", side_effect=Exception("MLflow Server Down!"))

@pytest.fixture
def client_healthy(mock_mlflow_success):
    """
    O 'with TestClient' é obrigatório no FastAPI moderno para acionar o 
    evento de 'lifespan' (que baixa o modelo no startup).
    """
    with TestClient(app) as client:
        yield client

@pytest.fixture
def client_unhealthy(mock_mlflow_failure):
    """Cliente que falhou ao inicializar por queda do MLflow."""
    with TestClient(app) as client:
        yield client

# ==========================================
# 3. Testes do Health Check (/health)
# ==========================================
def test_health_check_healthy(client_healthy):
    """Garante que a API avisa o Kubernetes que está pronta (200 OK)."""
    response = client_healthy.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_health_check_unhealthy(client_unhealthy):
    """Garante que a API devolve 503 se o MLflow falhar no startup."""
    response = client_unhealthy.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "unhealthy"

# ==========================================
# 4. Testes de Inferência (/predict/batch)
# ==========================================
def test_predict_batch_success(client_healthy):
    """Testa o fluxo completo: requisição -> feature eng -> predição -> resposta."""
    payload = {
        "predictions": [
            {
                "date": "2026-09-24T12:00:00Z",
                "wind_speed_100m": 12.5,
                "wind_direction_100m": 180.0,
                "temperature_2m": 25.0
            }
        ]
    }
    
    response = client_healthy.post("/predict/batch", json=payload)
    
    # A API deve responder 200 OK
    assert response.status_code == 200
    
    # Validação do cálculo matemático da resposta
    data = response.json()
    assert len(data) == 1
    # Como o FakeModel previu 0.5, esperamos que o FC seja 0.5
    assert data[0]["predicted_fc"] == 0.5
    # O predicted_mw não pode ser nulo se o pipeline de features funcionou
    assert data[0]["predicted_mw"] is not None 

def test_predict_batch_invalid_payload(client_healthy):
    """Garante que o FastAPI + Pydantic barram requisições malformadas (HTTP 422)."""
    payload_invalido = {
        "predictions": [
            {
                "date": "2026-09-24T12:00:00Z",
                "wind_speed_100m": -10.0, # Vento negativo (Proibido pelo Schema)
                "wind_direction_100m": 180.0,
                "temperature_2m": 25.0
            }
        ]
    }
    
    response = client_healthy.post("/predict/batch", json=payload_invalido)
    assert response.status_code == 422 # Unprocessable Entity