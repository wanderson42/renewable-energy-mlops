import os

import mlflow.tracking
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from prefect import flow, task

from energy_mlops.config import settings
from energy_mlops.models.interfaces import (
    ModelOptimizer,
    ModelTrainer,
    OptimizationResult,
)
from energy_mlops.models.optimize_stacking_ensemble import (
    run_optimization as default_stacking_optimizer,
)

# Importando a função de treino atual e a apelidamos para injetá-la como padrão
from energy_mlops.models.train_stacking_ensemble import (
    train_stacking_regressor as default_stacking_trainer,
)
from energy_mlops.pipelines.utils import save_dataset_to_lake_or_local

# Injeção global de credenciais S3/RustFS para boto3 e MLflow Artifacts
os.environ["AWS_ACCESS_KEY_ID"] = settings.RUSTFS_ROOT_USER
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.RUSTFS_ROOT_PASSWORD
os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.RUSTFS_ENDPOINT
os.environ["AWS_ENDPOINT_URL"] = settings.RUSTFS_ENDPOINT


# ==============================================================================
# PASSO 1: EXTRAÇÃO, PERSISTÊNCIA E PADRONIZAÇÃO DE DATASETS
# ==============================================================================
@task(name="1. Extrair e Salvar Janela Expansiva", retries=2, retry_delay_seconds=30)
def fetch_expanding_window_data() -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Extrai os dados aplicando a regra de MLOps de Mês Fechado (Lag M-1),
    persiste os datasets na camada Gold via utils.py e retorna DataFrames e nomes.
    """
    s3_gold_path = f"s3://{settings.RUSTFS_BUCKET}/gold/"
    logger.info(f"📖 Lendo e concatenando parquets da camada Gold: {s3_gold_path}")

    df = pd.read_parquet(s3_gold_path, storage_options=settings.storage_options)

    date_col = "date" if "date" in df.columns else "data"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)

    max_date = df[date_col].max()
    first_day_current_month = max_date.replace(day=1, hour=0, minute=0, second=0)
    current_month_hours = len(df[df[date_col] >= first_day_current_month])

    if current_month_hours < 28 * 24:
        logger.warning(
            f"⚠️ Mês recente ({first_day_current_month.strftime('%Y-%m')}) incompleto. "
            f"Deslocando janela para o último mês fechado."
        )
        oot_start_date = (first_day_current_month - pd.Timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0)
        oot_end_date = first_day_current_month
    else:
        oot_start_date = first_day_current_month
        oot_end_date = max_date

    df_train = df[df[date_col] < oot_start_date].copy()
    df_test = df[(df[date_col] >= oot_start_date) & (df[date_col] < oot_end_date)].copy()

    train_start_str = df_train[date_col].min().strftime("%Y_%m")
    cutoff_str = oot_start_date.strftime("%Y_%m")

    train_file_label = f"train_wind_energy_{train_start_str}_expanding_up_to_{cutoff_str}.parquet"
    test_file_label = f"oot_test_wind_energy_{cutoff_str}.parquet"

    save_dataset_to_lake_or_local(
        df=df_train,
        key=f"gold/{train_file_label}",
        local_path=os.path.join("data", "gold", train_file_label)
    )
    
    save_dataset_to_lake_or_local(
        df=df_test,
        key=f"gold/{test_file_label}",
        local_path=os.path.join("data", "gold", test_file_label)
    )

    logger.info("✅ Datasets de Treino e Teste processados e sincronizados com sucesso!")
    return df_train, df_test, train_file_label, test_file_label


# ==============================================================================
# PASSO 2: OTIMIZAÇÃO E TREINAMENTO AGNÓSTICO
# ==============================================================================
@task(name="2. Otimizar Hiperparâmetros (Agnóstico)", retries=1)
def optimize_hyperparameters(
    optimizer_func: ModelOptimizer | None,
    df_train: pd.DataFrame,
    train_file: str,
) -> OptimizationResult:
    if optimizer_func is None:
        raise ValueError(
            "O treinamento configurado exige uma execução de otimização, "
            "mas optimizer_func é None."
        )

    result = optimizer_func(
        df_train=df_train,
        train_file=train_file,
        n_trials=10,
        n_splits=3,
    )

    if not result:
        raise RuntimeError(
            "O otimizador não retornou um resultado válido."
        )

    if "optimization_run_id" not in result:
        raise RuntimeError(
            "Resultado da otimização não contém 'optimization_run_id'."
        )

    if "best_params" not in result:
        raise RuntimeError(
            "Resultado da otimização não contém 'best_params'."
        )

    logger.info(
        f"✅ Otimização concluída. "
        f"Run MLflow: {result['optimization_run_id']}"
    )

    return result


@task(name="3. Treinar Modelo (Agnóstico)", log_prints=True)
def execute_training(
    trainer_func: ModelTrainer,
    optimization_result: OptimizationResult,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    train_file: str,
    test_file: str,
) -> tuple[str, float]:
    """
    Executa o treinamento utilizando explicitamente os hiperparâmetros
    produzidos pela execução atual da otimização.
    """

    logger.info(
        f"⚙️ Iniciando treinamento utilizando: "
        f"{trainer_func.__name__}"
    )

    optimization_run_id = optimization_result["optimization_run_id"]
    best_params = optimization_result["best_params"]

    logger.info(
        f"🔗 Treinamento vinculado à Optimization Run: "
        f"{optimization_run_id}"
    )

    run_id, mae = trainer_func(
        df_train,
        df_test,
        train_file,
        test_file,
        best_params=best_params,
        optimization_run_id=optimization_run_id,
    )

    return run_id, mae


# ==============================================================================
# QUALITY GATE (NAVALHA DE OCKHAM: CHAMPION VS CHALLENGER)
# ==============================================================================

def get_registered_model_version(
    client: MlflowClient,
    model_name: str,
    run_id: str,
) -> str:
    """Obtém a versão registrada correspondente à run do Challenger."""

    versions = client.search_model_versions(
        filter_string=(
            f"name='{model_name}' "
            f"and run_id='{run_id}'"
        )
    )

    if not versions:
        raise RuntimeError(
            f"Nenhuma versão registrada encontrada para "
            f"model='{model_name}' e run_id='{run_id}'."
        )

    return max(
        versions,
        key=lambda version: int(version.version),
    ).version


@task(
    name="4. Quality Gate (Champion vs Challenger)",
    log_prints=True,
)
def evaluate_and_promote(
    challenger_run_id: str,
    challenger_mae: float,
):
    model_name = "ensemble_lgb_xgb_rf_bahia"
    client = mlflow.tracking.MlflowClient()

    challenger_run = client.get_run(challenger_run_id)

    challenger_nmae = float(
        challenger_run.data.metrics.get(
            "oot_nmae_pct_2026",
            float("inf"),
        )
    )

    challenger_features = int(
        challenger_run.data.params.get(
            "num_features",
            0,
        )
    )

    nmae_diff = None

    try:
        champion_info = client.get_model_version_by_alias(
            model_name,
            "champion",
        )

    except MlflowException:
        logger.warning(
            f"ℹ️ Nenhum '@champion' encontrado para "
            f"'{model_name}'."
        )

        promote_to_champion = True
        champion_mae = None
        champion_nmae = None
        champion_features = None

    else:
        champ_run = client.get_run(champion_info.run_id)

        champion_mae_raw = (
            champ_run.data.metrics.get("oot_mae_mw")
            or champ_run.data.metrics.get("oot_mae_mw_2026")
        )

        if champion_mae_raw is None:
            raise RuntimeError(
                f"Champion v{champion_info.version} não possui "
                "métrica 'oot_mae_mw' ou 'oot_mae_mw_2026'."
            )

        champion_mae = float(champion_mae_raw)

        champion_nmae = float(
            champ_run.data.metrics.get(
                "oot_nmae_pct_2026",
                float("inf"),
            )
        )

        champion_features = int(
            champ_run.data.params.get(
                "num_features",
                999,
            )
        )

        print(
            f"🏆 Champion atual "
            f"(v{champion_info.version}): "
            f"MAE={champion_mae:.2f} MW | "
            f"nMAE={champion_nmae:.2f}% | "
            f"Features={champion_features}"
        )

        print(
            f"⚔️ Challenger "
            f"(Run {challenger_run_id[:8]}): "
            f"MAE={challenger_mae:.2f} MW | "
            f"nMAE={challenger_nmae:.2f}% | "
            f"Features={challenger_features}"
        )

        mae_improved = challenger_mae < champion_mae
        fewer_features = challenger_features < champion_features

        nmae_diff = challenger_nmae - champion_nmae

        within_tolerance = (
            fewer_features
            and nmae_diff <= 0.05
        )

        promote_to_champion = (
            mae_improved
            or within_tolerance
        )

        if not promote_to_champion:
            reason = (
                f"MAE superior "
                f"({challenger_mae:.2f} vs "
                f"{champion_mae:.2f} MW)"
                if not mae_improved
                else (
                    f"Condição de simplificação não satisfeita: "
                    f"{challenger_features} vs "
                    f"{champion_features} features"
                )
            )

            logger.warning(
                f"❌ Challenger rejeitado: {reason}"
            )

            client.set_tag(
                challenger_run_id,
                "quality_gate_status",
                "REJECTED",
            )
            client.set_tag(
                challenger_run_id,
                "rejection_reason",
                reason,
            )

            return

    # ================================================================
    # PROMOÇÃO
    # ================================================================

    challenger_version = get_registered_model_version(
        client,
        model_name,
        challenger_run_id,
    )

    client.set_registered_model_alias(
        model_name,
        "champion",
        challenger_version,
    )

    client.set_tag(
        challenger_run_id,
        "quality_gate_status",
        "PROMOTED_CHAMPION",
    )

    client.set_tag(
        challenger_run_id,
        "quality_gate_champion_mae_mw",
        str(champion_mae),
    )

    client.set_tag(
        challenger_run_id,
        "N/A_FIRST_CHAMPION" if nmae_diff is None else str(nmae_diff),
        str(nmae_diff),
    )

    logger.info(
        f"🚀 Modelo v{challenger_version} promovido "
        f"para '@champion'."
    )

    # ================================================================
    # HOT RELOAD
    # ================================================================

    try:
        import requests

        api_reload_url = os.getenv(
            "API_RELOAD_URL",
            "http://localhost:8000/reload-model",
        )

        response = requests.post(
            api_reload_url,
            timeout=5,
        )

        response.raise_for_status()

        logger.info(
            "⚡ HOT-RELOAD: API atualizada com sucesso."
        )

    except requests.RequestException as exc:
        logger.warning(
            f"⚠️ Modelo promovido, mas o HOT-RELOAD falhou: {exc}"
        )

# ==========================================
# REGISTRY DE DEPENDÊNCIAS
# ==========================================
TRAINER_REGISTRY = {
    "stacking": default_stacking_trainer,
}

OPTIMIZER_REGISTRY = {
    "stacking": default_stacking_optimizer,
    "none": None
}

# ==============================================================================
# FLUXO PRINCIPAL PREFECT (ORQUESTRADOR CT)
# ==============================================================================
@flow(name="Pipeline de Treinamento Contínuo - Energia Eólica Bahia")
def continuous_training_pipeline(
    trainer_name: str = "stacking",
    optimizer_name: str = "stacking",
):
    """
    Orquestrador agnóstico de treinamento contínuo.

    A execução atual da otimização produz explicitamente os parâmetros
    consumidos pela execução atual do treinamento.
    """

    trainer_algorithm = TRAINER_REGISTRY.get(trainer_name)
    optimizer_algorithm = OPTIMIZER_REGISTRY.get(optimizer_name)

    if trainer_algorithm is None:
        raise ValueError(
            f"Trainer '{trainer_name}' não encontrado no Registry."
        )

    logger.info(
        "🚀 Iniciando Pipeline de Treinamento Contínuo (CT)..."
    )

    # 1. Dados
    df_train, df_test, train_file, test_file = (
        fetch_expanding_window_data()
    )

    # 2. Otimização
    optimization_result = optimize_hyperparameters(
        optimizer_func=optimizer_algorithm,
        df_train=df_train,
        train_file=train_file,
    )

    # 3. Treinamento usando EXPLICITAMENTE o resultado acima
    challenger_run_id, challenger_mae = execute_training(
        trainer_func=trainer_algorithm,
        optimization_result=optimization_result,
        df_train=df_train,
        df_test=df_test,
        train_file=train_file,
        test_file=test_file,
    )

    # 4. Quality Gate
    evaluate_and_promote(
        challenger_run_id,
        challenger_mae,
    )

    logger.info(
        "🏁 Pipeline de Treinamento Contínuo finalizado com sucesso!"
    )


if __name__ == "__main__":
    continuous_training_pipeline()