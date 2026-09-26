# Testes de unidade, variáveis de ambiente e Pydantic Settings: src/energy_mlops/config.py
from energy_mlops.config import Settings


def test_settings_default_values():
    """Verifica se os valores padrão ou carregados do ambiente são válidos."""
    settings = Settings()
    
    assert settings.RUSTFS_ENDPOINT.startswith("http")
    assert isinstance(settings.RUSTFS_ROOT_USER, str)
    assert isinstance(settings.MLFLOW_TRACKING_URI, str)


def test_storage_options_property():
    """Testa a geração do dicionário de configuração S3FS para o Pandas/RustFS."""
    settings = Settings(
        RUSTFS_ENDPOINT="http://localhost:9000",
        RUSTFS_ROOT_USER="test_user",
        RUSTFS_ROOT_PASSWORD="test_password"
    )

    storage_opts = settings.storage_options

    assert storage_opts["key"] == "test_user"
    assert storage_opts["secret"] == "test_password"
    assert storage_opts["client_kwargs"]["endpoint_url"] == "http://localhost:9000"


def test_settings_environment_override(monkeypatch):
    """Garante que variáveis injetadas via SO (ex: Kubernetes) sobrepõem os padrões."""
    monkeypatch.setenv("RUSTFS_ENDPOINT", "http://k8s-rustfs-service:9000")
    monkeypatch.setenv("RUSTFS_ROOT_USER", "k8s_user")

    settings = Settings()

    assert settings.RUSTFS_ENDPOINT == "http://k8s-rustfs-service:9000"
    assert settings.RUSTFS_ROOT_USER == "k8s_user"
