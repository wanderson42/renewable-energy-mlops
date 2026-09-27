import os

import mlflow.tracking
import pandas as pd
from loguru import logger
from mlflow.exceptions import MlflowException
from prefect import flow, task

from energy_mlops.config import settings
from energy_mlops.models.interfaces import ModelOptimizer, ModelTrainer
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
    train_file: str
) -> dict:
    if optimizer_func is None:
        logger.info("⏩ Etapa de otimização ignorada para este modelo.")
        return {}
    return optimizer_func(df_train=df_train, train_file=train_file, n_trials=10, n_splits=3)

@task(name="3. Treinar Modelo (Agnóstico)", log_prints=True)
def execute_training(
    trainer_func: ModelTrainer, 
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    train_file: str, 
    test_file: str
) -> tuple[str, float]:
    """
    Recebe qualquer algoritmo de treino via injeção de dependência e executa.
    A task não sabe se é um XGBoost, Stacking ou Rede Neural.
    """
    logger.info(f"⚙️ Iniciando treinamento utilizando o algoritmo fornecido: {trainer_func.__name__}")
    run_id, mae = trainer_func(df_train, df_test, train_file, test_file)
    return run_id, mae


# ==============================================================================
# PASSO 3: QUALITY GATE (NAVALHA DE OCKHAM: CHAMPION VS CHALLENGER)
# ==============================================================================
@task(name="4. Quality Gate (Champion vs Challenger)", log_prints=True)
def evaluate_and_promote(challenger_run_id: str, challenger_mae: float):
    model_name = "ensemble_lgb_xgb_rf_bahia"
    client = mlflow.tracking.MlflowClient()

    print("\n🛡️ INICIANDO QUALITY GATE NA ARENA (Champion vs Challenger)...")

    promote_to_champion = False
    challenger_run = client.get_run(challenger_run_id)
    challenger_nmae = float(challenger_run.data.metrics.get("oot_nmae_pct_2026", 0.0))
    challenger_features = int(challenger_run.data.params.get("num_features", 0))

    try:
        champion_info = client.get_model_version_by_alias(model_name, "champion")
        champ_run = client.get_run(champion_info.run_id)

        champ_mae = champ_run.data.metrics.get("oot_mae_mw") or champ_run.data.metrics.get("oot_mae_mw_2026")
        champ_nmae = float(champ_run.data.metrics.get("oot_nmae_pct_2026", 0.0))
        champ_features = int(champ_run.data.params.get("num_features", 999))

        print(f"🏆 Champion Atual  (v{champion_info.version}): MAE = {champ_mae:.2f} MW | nMAE = {champ_nmae:.2f}% | Features = {champ_features}")
        print(f"⚔️ Challenger Novo (Run {challenger_run_id[:8]}): MAE = {challenger_mae:.2f} MW | nMAE = {challenger_nmae:.2f}% | Features = {challenger_features}")

        mae_improved = challenger_mae < champ_mae
        fewer_features = challenger_features < champ_features
        nmae_diff = challenger_nmae - champ_nmae
        within_tolerance = fewer_features and (nmae_diff <= 0.05)

        if mae_improved:
            gain = champ_mae - challenger_mae
            print(f"🎉 APROVADO POR DESEMPENHO! Redução de {gain:.2f} MW no MAE.")
            promote_to_champion = True
        elif within_tolerance:
            print(
                f"🎉 APROVADO PELA NAVALHA DE OCKHAM! Arquitetura mais simples "
                f"({challenger_features} vs {champ_features} features) com variação aceitável no nMAE "
                f"({nmae_diff:+.3f}% <= +0.05%)."
            )
            promote_to_champion = True
        else:
            reason = (
                f"MAE superior ({challenger_mae:.2f} vs {champ_mae:.2f} MW)"
                if not fewer_features
                else f"Aumento no nMAE ({nmae_diff:+.3f}%) acima da tolerância (+0.05%)"
            )
            print(f"❌ REPROVADO! {reason}. Mantendo produção intacta.")
            client.set_tag(challenger_run_id, "quality_gate_status", "REJECTED")
            client.set_tag(challenger_run_id, "rejection_reason", reason)

    except MlflowException:
        print("ℹ️ Nenhum modelo rotulado como '@champion' foi encontrado. Promovendo o Challenger como Primeiro Campeão!")
        promote_to_champion = True

    if promote_to_champion:
        runs = client.search_model_versions(f"name='{model_name}'")
        latest_version = max([int(m.version) for m in runs if m.run_id == challenger_run_id], default=1)

        client.set_registered_model_alias(model_name, "champion", str(latest_version))
        client.set_tag(challenger_run_id, "quality_gate_status", "PROMOTED_CHAMPION")
        print(f"🚀 SUCESSO: Modelo versão {latest_version} promovido para '@champion' no Registry!")

        try:
            import requests
            api_reload_url = os.getenv("API_RELOAD_URL", "http://localhost:8000/reload-model")
            response = requests.post(api_reload_url, timeout=5)
            if response.status_code == 200:
                print("⚡ HOT-RELOAD: API de Inferência notificada e atualizada com sucesso sem downtime!")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ AVISO: Não foi possível notificar a API para Hot-Reload automático: {e}")


# ==============================================================================
# FLUXO PRINCIPAL PREFECT (ORQUESTRADOR CT)
# ==============================================================================
@flow(name="Pipeline de Treinamento Contínuo - Energia Eólica Bahia")
def continuous_training_pipeline(
    trainer_algorithm: ModelTrainer = default_stacking_trainer,
    optimizer_algorithm: ModelOptimizer | None = default_stacking_optimizer
):
    """
    Orquestrador Agnóstico. Por padrão, utiliza o Stacking Ensemble, mas 
    pode receber qualquer algoritmo que respeite o TrainerContract.
    """
    logger.info("🚀 Iniciando Pipeline de Treinamento Contínuo (CT)...")

    df_train, df_test, train_file, test_file = fetch_expanding_window_data()

    # Otimização dinamica ou ignorada dependendo do modelo injetado
    _ = optimize_hyperparameters(
        optimizer_func=optimizer_algorithm, 
        df_train=df_train, 
        train_file=train_file
    )

    # Injeção de Dependência na task de treinamento
    challenger_run_id, challenger_mae = execute_training(
        trainer_func=trainer_algorithm,
        df_train=df_train, 
        df_test=df_test, 
        train_file=train_file, 
        test_file=test_file
    )

    evaluate_and_promote(challenger_run_id, challenger_mae)

    logger.info("🏁 Pipeline de Treinamento Contínuo finalizado com sucesso!")


if __name__ == "__main__":
    continuous_training_pipeline()