import os
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd
import requests
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
from energy_mlops.models.train_stacking_ensemble import (
    train_stacking_regressor as default_stacking_trainer,
)
from energy_mlops.pipelines.utils import save_dataset_to_lake_or_local

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

MODEL_NAME = "ensemble_lgb_xgb_rf_bahia"
MODEL_ALIAS = "champion"

# Um Challenger com menos features pode ser promovido mesmo sem melhorar o MAE,
# desde que a degradação de nMAE não ultrapasse esta tolerância.
NMAE_SIMPLIFICATION_TOLERANCE_PP = 0.05


# Injeção global de credenciais S3/RustFS para boto3 e MLflow Artifacts.
os.environ["AWS_ACCESS_KEY_ID"] = settings.RUSTFS_ROOT_USER
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.RUSTFS_ROOT_PASSWORD
os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.RUSTFS_ENDPOINT
os.environ["AWS_ENDPOINT_URL"] = settings.RUSTFS_ENDPOINT


# ==============================================================================
# HELPERS DE CONTRATO DE MÉTRICAS
# ==============================================================================


def resolve_metric(
    metrics: Mapping[str, Any],
    canonical_name: str,
    *,
    required: bool = True,
) -> float | None:
    """
    Resolve uma métrica MLflow priorizando o nome canônico.

    Compatibilidade histórica:
    Runs antigas podem possuir métricas com sufixo anual, por exemplo:

        oot_nmae_pct_YYYY

    O ano não faz parte do contrato atual. Portanto:

    1. procura primeiro a chave canônica;
    2. se ausente, procura uma única variante histórica ``_<ano>``;
    3. se houver mais de uma variante histórica, falha explicitamente
       para evitar seleção silenciosa de uma métrica ambígua;
    4. se nenhuma métrica existir e ``required=True``, falha.

    Exemplos aceitos:
        oot_nmae_pct
        oot_nmae_pct_2024
        oot_nmae_pct_2025
        oot_nmae_pct_2026

    Nenhum ano específico é codificado na lógica.
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

        match = legacy_pattern.fullmatch(
            metric_name
        )

        if match is not None:
            legacy_candidates.append(
                (
                    metric_name,
                    metric_value,
                )
            )

    if len(legacy_candidates) == 1:
        metric_name, metric_value = (
            legacy_candidates[0]
        )

        logger.warning(
            "⚠️ Usando métrica legada "
            f"'{metric_name}' como fallback para "
            f"'{canonical_name}'."
        )

        return float(metric_value)

    if len(legacy_candidates) > 1:
        candidates = ", ".join(
            name
            for name, _ in legacy_candidates
        )

        raise RuntimeError(
            f"Métrica '{canonical_name}' ausente e "
            "foram encontradas múltiplas variantes "
            f"legadas: {candidates}. "
            "A seleção seria ambígua."
        )

    if required:
        raise RuntimeError(
            "Métrica obrigatória "
            f"'{canonical_name}' não encontrada."
        )

    return None


def get_required_num_features(
    params: Mapping[str, Any],
    *,
    role: str,
) -> int:
    """
    Obtém o número de features registrado na Run.

    O Challenger deve sempre possuir este parâmetro,
    pois ele é necessário para a regra de parcimônia.
    """

    value = params.get(
        "num_features"
    )

    if value is None:
        raise RuntimeError(
            f"{role} não possui o parâmetro "
            "'num_features'."
        )

    return int(value)


# ==============================================================================
# PASSO 1: EXTRAÇÃO, PERSISTÊNCIA E PADRONIZAÇÃO DE DATASETS
# ==============================================================================


@task(
    name="1. Extrair e Salvar Janela Expansiva",
    retries=2,
    retry_delay_seconds=30,
)
def fetch_expanding_window_data(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    str,
    str,
]:
    """
    Extrai os dados aplicando a regra de MLOps de Mês Fechado (Lag M-1),
    persiste os datasets na camada Gold via utils.py e retorna
    DataFrames e nomes.
    """

    s3_gold_path = (
        f"s3://{settings.RUSTFS_BUCKET}/gold/"
    )

    logger.info(
        "📖 Lendo e concatenando parquets "
        f"da camada Gold: {s3_gold_path}"
    )

    df = pd.read_parquet(
        s3_gold_path,
        storage_options=settings.storage_options,
    )

    date_col = (
        "date"
        if "date" in df.columns
        else "data"
    )

    df[date_col] = pd.to_datetime(
        df[date_col]
    )

    df = (
        df.sort_values(
            by=date_col
        )
        .reset_index(
            drop=True
        )
    )

    max_date = df[date_col].max()

    first_day_current_month = max_date.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
    )

    current_month_hours = len(
        df[
            df[date_col]
            >= first_day_current_month
        ]
    )

    if current_month_hours < 28 * 24:
        logger.warning(
            "⚠️ Mês recente "
            f"({first_day_current_month.strftime('%Y-%m')}) "
            "incompleto. Deslocando janela para o "
            "último mês fechado."
        )

        oot_start_date = (
            first_day_current_month
            - pd.Timedelta(days=1)
        ).replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
        )

        oot_end_date = (
            first_day_current_month
        )

    else:
        oot_start_date = (
            first_day_current_month
        )

        oot_end_date = max_date

    df_train = df[
        df[date_col] < oot_start_date
    ].copy()

    df_test = df[
        (
            df[date_col]
            >= oot_start_date
        )
        & (
            df[date_col]
            < oot_end_date
        )
    ].copy()

    train_start_str = (
        df_train[date_col]
        .min()
        .strftime("%Y_%m")
    )

    cutoff_str = (
        oot_start_date
        .strftime("%Y_%m")
    )

    train_file_label = (
        f"train_wind_energy_{train_start_str}"
        f"_expanding_up_to_{cutoff_str}.parquet"
    )

    test_file_label = (
        f"oot_test_wind_energy_"
        f"{cutoff_str}.parquet"
    )

    save_dataset_to_lake_or_local(
        df=df_train,
        key=(
            f"gold/"
            f"{train_file_label}"
        ),
        local_path=os.path.join(
            "data",
            "gold",
            train_file_label,
        ),
    )

    save_dataset_to_lake_or_local(
        df=df_test,
        key=(
            f"gold/"
            f"{test_file_label}"
        ),
        local_path=os.path.join(
            "data",
            "gold",
            test_file_label,
        ),
    )

    logger.info(
        "✅ Datasets de Treino e Teste "
        "processados e sincronizados com sucesso!"
    )

    return (
        df_train,
        df_test,
        train_file_label,
        test_file_label,
    )


# ==============================================================================
# PASSO 2: OTIMIZAÇÃO E TREINAMENTO AGNÓSTICO
# ==============================================================================


@task(
    name="2. Otimizar Hiperparâmetros (Agnóstico)",
    retries=1,
)
def optimize_hyperparameters(
    optimizer_func: ModelOptimizer,
    df_train: pd.DataFrame,
    train_file: str,
) -> OptimizationResult:
    """
    Executa uma estratégia de otimização registrada.

    Esta task somente é chamada quando existe um optimizer configurado.
    A ausência deliberada de otimização é tratada pelo orquestrador.
    """

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
            "Resultado da otimização não contém "
            "'optimization_run_id'."
        )

    if "best_params" not in result:
        raise RuntimeError(
            "Resultado da otimização não contém "
            "'best_params'."
        )

    logger.info(
        "✅ Otimização concluída. "
        f"Run MLflow: {result['optimization_run_id']}"
    )

    return result


@task(
    name="3. Treinar Modelo (Agnóstico)",
    log_prints=True,
)
def execute_training(
    trainer_func: ModelTrainer,
    optimization_result: OptimizationResult | None,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    train_file: str,
    test_file: str,
) -> tuple[str, float]:
    """
    Executa um trainer agnóstico.

    Se houver uma etapa de otimização anterior, seus parâmetros
    e sua Run MLflow são propagados para o trainer.

    Trainers que não exigem otimização recebem ``None`` para
    ``best_params`` e ``optimization_run_id``.
    """

    logger.info(
        "⚙️ Iniciando treinamento utilizando: "
        f"{trainer_func.__name__}"
    )

    if optimization_result is None:
        best_params = None
        optimization_run_id = None

        logger.info(
            "ℹ️ Treinamento configurado sem etapa "
            "de otimização."
        )

    else:
        best_params = optimization_result[
            "best_params"
        ]

        optimization_run_id = (
            optimization_result[
                "optimization_run_id"
            ]
        )

        logger.info(
            "🔗 Treinamento vinculado à Optimization Run: "
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
# PASSO 3: QUALITY GATE — CHAMPION VS CHALLENGER
# ==============================================================================


def get_registered_model_version(
    client: MlflowClient,
    model_name: str,
    run_id: str,
) -> str:
    """
    Obtém a versão registrada correspondente
    à Run do Challenger.
    """

    versions = (
        client.search_model_versions(
            filter_string=(
                f"name='{model_name}' "
                f"and run_id='{run_id}'"
            )
        )
    )

    if not versions:
        raise RuntimeError(
            "Nenhuma versão registrada "
            "encontrada para "
            f"model='{model_name}' e "
            f"run_id='{run_id}'."
        )

    return max(
        versions,
        key=lambda version: int(
            version.version
        ),
    ).version


@task(
    name="4. Quality Gate (Champion vs Challenger)",
    log_prints=True,
)
def evaluate_and_promote(
    challenger_run_id: str,
    challenger_mae: float,
):
    """
    Compara Challenger e Champion.

    Regra de promoção:
    - promove se o MAE do Challenger melhorar;
    - ou, pela regra de parcimônia, se usar menos
      features e a degradação de nMAE for no máximo
      NMAE_SIMPLIFICATION_TOLERANCE_PP.

    Métricas atuais usam nomes canônicos.
    Runs históricas com sufixo anual são aceitas
    apenas por meio do resolvedor de compatibilidade.
    """

    client = MlflowClient(
        tracking_uri=(
            settings.MLFLOW_TRACKING_URI
        )
    )

    challenger_run = client.get_run(
        challenger_run_id
    )

    challenger_nmae = resolve_metric(
        challenger_run.data.metrics,
        "oot_nmae_pct",
    )

    challenger_features = (
        get_required_num_features(
            challenger_run.data.params,
            role="Challenger",
        )
    )

    nmae_diff = None

    try:
        champion_info = (
            client.get_model_version_by_alias(
                MODEL_NAME,
                MODEL_ALIAS,
            )
        )

    except MlflowException:
        logger.warning(
            "ℹ️ Nenhum '@champion' encontrado "
            f"para '{MODEL_NAME}'."
        )

        promotion_reason = (
            "FIRST_CHAMPION"
        )

        champion_mae = None
        champion_nmae = None
        champion_features = None

    else:
        champion_run = client.get_run(
            champion_info.run_id
        )

        champion_mae = resolve_metric(
            champion_run.data.metrics,
            "oot_mae_mw",
        )

        champion_nmae = resolve_metric(
            champion_run.data.metrics,
            "oot_nmae_pct",
        )

        champion_features_raw = (
            champion_run.data.params.get(
                "num_features"
            )
        )

        champion_features = (
            int(champion_features_raw)
            if champion_features_raw is not None
            else None
        )

        print(
            "🏆 Champion atual "
            f"(v{champion_info.version}): "
            f"MAE={champion_mae:.2f} MW | "
            f"nMAE={champion_nmae:.2f}% | "
            f"Features={champion_features}"
        )

        print(
            "⚔️ Challenger "
            f"(Run {challenger_run_id[:8]}): "
            f"MAE={challenger_mae:.2f} MW | "
            f"nMAE={challenger_nmae:.2f}% | "
            f"Features={challenger_features}"
        )

        mae_improved = (
            challenger_mae < champion_mae
        )

        fewer_features = (
            champion_features is not None
            and challenger_features
            < champion_features
        )

        nmae_diff = (
            challenger_nmae
            - champion_nmae
        )

        within_tolerance = (
            fewer_features
            and nmae_diff
            <= NMAE_SIMPLIFICATION_TOLERANCE_PP
        )

        if mae_improved:
            promotion_reason = (
                "MAE_IMPROVED"
            )

        elif within_tolerance:
            promotion_reason = (
                "PARSIMONY_WITHIN_"
                "NMAE_TOLERANCE"
            )

        else:
            if champion_features is None:
                reason = (
                    "MAE não melhorou "
                    f"({challenger_mae:.2f} vs "
                    f"{champion_mae:.2f} MW) e "
                    "o Champion não possui "
                    "'num_features' para avaliar "
                    "a regra de parcimônia."
                )

            elif not fewer_features:
                reason = (
                    "MAE não melhorou "
                    f"({challenger_mae:.2f} vs "
                    f"{champion_mae:.2f} MW) e "
                    "o Challenger não reduz "
                    "o número de features "
                    f"({challenger_features} vs "
                    f"{champion_features})."
                )

            else:
                reason = (
                    "O Challenger usa menos "
                    "features, mas a diferença "
                    "de nMAE excede a tolerância: "
                    f"{nmae_diff:+.4f} p.p. > "
                    f"{NMAE_SIMPLIFICATION_TOLERANCE_PP:.4f} "
                    "p.p."
                )

            logger.warning(
                "❌ Challenger rejeitado: "
                f"{reason}"
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

            client.set_tag(
                challenger_run_id,
                "quality_gate_nmae_delta_pp",
                str(nmae_diff),
            )

            return

    # ==========================================================================
    # PROMOÇÃO
    # ==========================================================================

    challenger_version = (
        get_registered_model_version(
            client,
            MODEL_NAME,
            challenger_run_id,
        )
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        MODEL_ALIAS,
        challenger_version,
    )

    client.set_tag(
        challenger_run_id,
        "quality_gate_status",
        "PROMOTED_CHAMPION",
    )

    client.set_tag(
        challenger_run_id,
        "quality_gate_decision_reason",
        promotion_reason,
    )

    client.set_tag(
        challenger_run_id,
        "quality_gate_champion_mae_mw",
        (
            "N/A_FIRST_CHAMPION"
            if champion_mae is None
            else str(champion_mae)
        ),
    )

    client.set_tag(
        challenger_run_id,
        "quality_gate_nmae_delta_pp",
        (
            "N/A_FIRST_CHAMPION"
            if nmae_diff is None
            else str(nmae_diff)
        ),
    )

    logger.info(
        f"🚀 Modelo v{challenger_version} "
        "promovido para '@champion'."
    )

    # ==========================================================================
    # HOT RELOAD
    # ==========================================================================

    api_reload_url = os.getenv(
        "API_RELOAD_URL",
        "http://localhost:8000/reload-model",
    )

    try:
        response = requests.post(
            api_reload_url,
            timeout=5,
        )

        response.raise_for_status()

        logger.info(
            "⚡ HOT-RELOAD: API atualizada "
            "com sucesso."
        )

    except requests.RequestException as exc:
        logger.warning(
            "⚠️ Modelo promovido, mas o "
            f"HOT-RELOAD falhou: {exc}"
        )


# ==============================================================================
# REGISTRY DE DEPENDÊNCIAS
# ==============================================================================


TRAINER_REGISTRY = {
    "stacking": default_stacking_trainer,
}


OPTIMIZER_REGISTRY = {
    "stacking": default_stacking_optimizer,
    "none": None,
}


# ==============================================================================
# FLUXO PRINCIPAL PREFECT — ORQUESTRADOR CT
# ==============================================================================


@flow(
    name=(
        "Pipeline de Treinamento Contínuo "
        "- Energia Eólica Bahia"
    )
)
def continuous_training_pipeline(
    trainer_name: str = "stacking",
    optimizer_name: str = "stacking",
):
    """
    Orquestrador agnóstico de treinamento contínuo.

    O trainer é obrigatório. A etapa de otimização é opcional:

    - um optimizer registrado executa antes do treinamento;
    - ``optimizer_name="none"`` pula explicitamente essa etapa;
    - nomes desconhecidos falham antes da execução do pipeline.

    Quando não há otimização, o trainer recebe ``None`` em
    ``best_params`` e ``optimization_run_id``.
    """

    trainer_algorithm = (
        TRAINER_REGISTRY.get(
            trainer_name
        )
    )

    if trainer_algorithm is None:
        raise ValueError(
            f"Trainer '{trainer_name}' "
            "não encontrado no Registry."
        )

    # Diferencia explicitamente:
    #
    # "none"      -> ausência intencional de otimização
    # "stacking"  -> optimizer registrado
    # outro nome  -> erro de configuração
    if optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Optimizer '{optimizer_name}' "
            "não encontrado no Registry."
        )

    optimizer_algorithm = (
        OPTIMIZER_REGISTRY[
            optimizer_name
        ]
    )

    logger.info(
        "🚀 Iniciando Pipeline de "
        "Treinamento Contínuo (CT)..."
    )

    # ==========================================================================
    # 1. DADOS
    # ==========================================================================

    (
        df_train,
        df_test,
        train_file,
        test_file,
    ) = fetch_expanding_window_data()

    # ==========================================================================
    # 2. OTIMIZAÇÃO OPCIONAL
    # ==========================================================================

    if optimizer_algorithm is None:
        optimization_result = None

        logger.info(
            "ℹ️ Pipeline configurado sem etapa "
            "de otimização."
        )

    else:
        optimization_result = (
            optimize_hyperparameters(
                optimizer_func=(
                    optimizer_algorithm
                ),
                df_train=df_train,
                train_file=train_file,
            )
        )

    # ==========================================================================
    # 3. TREINAMENTO
    # ==========================================================================

    (
        challenger_run_id,
        challenger_mae,
    ) = execute_training(
        trainer_func=trainer_algorithm,
        optimization_result=(
            optimization_result
        ),
        df_train=df_train,
        df_test=df_test,
        train_file=train_file,
        test_file=test_file,
    )

    # ==========================================================================
    # 4. QUALITY GATE
    # ==========================================================================

    evaluate_and_promote(
        challenger_run_id,
        challenger_mae,
    )

    logger.info(
        "🏁 Pipeline de Treinamento "
        "Contínuo finalizado com sucesso!"
    )


if __name__ == "__main__":
    continuous_training_pipeline()
