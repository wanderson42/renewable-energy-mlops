from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Indica para o Pydantic buscar o arquivo .env na raiz
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignora variáveis no .env que não estejam declaradas na classe
    )

    # Credenciais e Endpoint do RustFS
    RUSTFS_BUCKET: str = "energy-lake"
    RUSTFS_ENDPOINT: str = "http://localhost:9000"
    RUSTFS_ROOT_USER: str
    RUSTFS_ROOT_PASSWORD: str 

    # Configurações do MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    # Configurações do Prefect
    PREFECT_API_URL: str = "http://localhost:4200/api"

    # Injeção automática das credenciais para leitura no Pandas/S3FS
    @property
    def storage_options(self) -> dict[str, str]:
        """Retorna o dicionário de opções de autenticação S3 para o Pandas/s3fs."""
        return {
            "key": self.RUSTFS_ROOT_USER,
            "secret": self.RUSTFS_ROOT_PASSWORD,
            "client_kwargs": {"endpoint_url": self.RUSTFS_ENDPOINT},
        }


settings = Settings()
