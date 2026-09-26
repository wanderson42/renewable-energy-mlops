import logging
import os
import socket
from urllib.parse import urlparse

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from energy_mlops.config import settings

# Configuração de logger para os utilitários
logger = logging.getLogger(__name__)

def check_rustfs_connection(endpoint_url: str = "http://localhost:9000", timeout: float = 1.0) -> bool:
    """Testa se a porta do RustFS está aberta e aceitando conexões."""
    try:
        parsed = urlparse(endpoint_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 9000
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False

def ensure_rustfs_bucket_exists(bucket_name: str) -> None:
    """Verifica se o bucket existe no RustFS e cria-o automaticamente caso não exista."""
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.RUSTFS_ENDPOINT,
        aws_access_key_id=settings.RUSTFS_ROOT_USER,
        aws_secret_access_key=settings.RUSTFS_ROOT_PASSWORD,
    )
    
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' já está pronto no RustFS.")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            logger.info(f"Bucket '{bucket_name}' não existe. A criar automaticamente...")
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            logger.error(f"Erro inesperado ao verificar o bucket {bucket_name}: {e}")
            raise e  # noqa: TRY201



def save_dataset_to_lake_or_local(df: pd.DataFrame, key: str, local_path: str) -> str:
    """
    Salva o DataFrame diretamente na camada Gold do RustFS usando o suporte nativo
    do Pandas, com fallback automático para o disco local em caso de falha.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_path = f"s3://{settings.RUSTFS_BUCKET}/{key}"

    # Dicionário de configuração compatível com o s3fs do Pandas para o RustFS
    storage_options = {
        "client_kwargs": {
            "endpoint_url": settings.RUSTFS_ENDPOINT,
        },
        "key": settings.RUSTFS_ROOT_USER,
        "secret": settings.RUSTFS_ROOT_PASSWORD,
        "config_kwargs": {
            "signature_version": "s3v4",
            "s3": {"addressing_style": "path"}
        }
    }

    try:
        logger.info(f"Tentando salvar dataset diretamente no RustFS: {s3_path}")
        
        # O Pandas gerencia o envio para o S3 de forma otimizada
        df.to_parquet(s3_path, index=False, storage_options=storage_options)
        
        logger.info(f"Dataset enviado com sucesso para a camada Gold do RustFS: {s3_path}")
        return s3_path

    except Exception as e:  # noqa: BLE001
        logger.warning(f"Falha na escrita no RustFS ({e}). Salvando localmente como fallback...")
        df.to_parquet(local_path, index=False)
        logger.info(f"Arquivo salvo com sucesso localmente em: {local_path}")
        return local_path