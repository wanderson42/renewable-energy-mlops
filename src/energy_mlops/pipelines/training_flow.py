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
from energy_mlops.data.snapshot_validation import audit_gold_snapshot
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

"""
Gold consolidado explícito
        ↓
load_gold_snapshot_task
        ↓
audit_gold_snapshot_task
        ↓
split_expanding_window_task
        ↓
split_expanding_window_snapshot   ← única regra de split
        ↓
   ┌─────────────┐
 TRAIN           OOT
   ↓              ↓
 audit          audit
   └──────┬──────┘
          ↓
persist_training_datasets_task
          ↓
Optuna
          ↓
Training
          ↓
Quality Gate
"""

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

MODEL_NAME = "ensemble_lgb_xgb_rf_bahia"
MODEL_ALIAS = "champion"
LOCAL_TIMEZONE = "America/Sao_Paulo"

DEFAULT_TRAINING_GOLD_SNAPSHOT_PATH = os.getenv(
    "TRAINING_GOLD_SNAPSHOT_PATH",
    (
        f"s3://{settings.RUSTFS_BUCKET}/gold/"
        "dataset_renewable_energy_2024_03_2026_08.parquet"
    ),
)

DEFAULT_OOT_YEAR = int(
    os.getenv(
        "TRAINING_OOT_YEAR",
        "2026",
    )
)

DEFAULT_OOT_MONTH = int(
    os.getenv(
        "TRAINING_OOT_MONTH",
        "8",
    )
)


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
# PASSO 1: CARREGAMENTO, AUDITORIA, SPLIT E PERSISTÊNCIA DOS DATASETS
# ==============================================================================


def get_local_month_bounds_utc(
    year: int,
    month: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Retorna os limites UTC correspondentes a um
    mês civil no timezone operacional da Bahia.

    Os timestamps retornados são timezone-naive,
    seguindo o contrato atual dos datasets Gold.
    """

    start_local = pd.Timestamp(
        year=year,
        month=month,
        day=1,
        tz=LOCAL_TIMEZONE,
    )

    end_local = (
        start_local
        + pd.offsets.MonthBegin(1)
    )

    start_utc = (
        start_local
        .tz_convert("UTC")
        .tz_localize(None)
    )

    end_utc = (
        end_local
        .tz_convert("UTC")
        .tz_localize(None)
    )

    return start_utc, end_utc


def split_expanding_window_snapshot(
    df: pd.DataFrame,
    *,
    oot_year: int,
    oot_month: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide um Gold consolidado em treino expansivo
    e um único mês civil OOT.

    O mês é definido em America/Sao_Paulo e depois
    convertido para UTC.
    """

    if "date" not in df.columns:
        raise ValueError(
            "Dataset sem coluna obrigatória 'date'."
        )

    result = df.copy()

    dates = pd.to_datetime(
        result["date"],
        errors="raise",
    )

    if dates.dt.tz is not None:
        dates = (
            dates
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )

    result["date"] = dates

    result = (
        result
        .sort_values("date")
        .reset_index(drop=True)
    )

    oot_start, oot_end = (
        get_local_month_bounds_utc(
            oot_year,
            oot_month,
        )
    )

    expected_last_hour = (
        oot_end
        - pd.Timedelta(hours=1)
    )

    max_date = result["date"].max()

    if max_date < expected_last_hour:
        raise ValueError(
            "Snapshot não contém o mês OOT "
            "completo. "
            f"Última hora disponível: {max_date}. "
            "Última hora necessária: "
            f"{expected_last_hour}."
        )

    df_train = (
        result.loc[
            result["date"] < oot_start
        ]
        .copy()
        .reset_index(drop=True)
    )

    df_test = (
        result.loc[
            (result["date"] >= oot_start)
            & (result["date"] < oot_end)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if df_train.empty:
        raise ValueError(
            "Snapshot de treino vazio."
        )

    if df_test.empty:
        raise ValueError(
            "Snapshot OOT vazio."
        )

    if (
        df_train["date"].max()
        >= df_test["date"].min()
    ):
        raise ValueError(
            "Overlap temporal entre treino e OOT."
        )

    return df_train, df_test


@task(
    name="Carregar Snapshot Gold",
    retries=2,
    retry_delay_seconds=30,
)
def load_gold_snapshot_task(
    snapshot_path: str,
) -> pd.DataFrame:
    """
    Carrega um único snapshot Gold explicitamente.

    Não lê o prefixo inteiro ``gold/`` para evitar
    concatenação acidental entre snapshots consolidados
    e artefatos derivados de treino/OOT.
    """

    normalized_path = snapshot_path.strip()

    if not normalized_path:
        raise ValueError(
            "snapshot_path não pode ser vazio."
        )

    if normalized_path.endswith("/"):
        raise ValueError(
            "snapshot_path deve apontar para um "
            "arquivo Parquet específico, não para "
            "um prefixo/diretório."
        )

    logger.info(
        "📖 Carregando snapshot Gold explícito: "
        f"{normalized_path}"
    )

    if normalized_path.startswith(
        "s3://"
    ):
        df = pd.read_parquet(
            normalized_path,
            storage_options=(
                settings.storage_options
            ),
        )
    else:
        df = pd.read_parquet(
            normalized_path
        )

    if df.empty:
        raise ValueError(
            "Snapshot Gold carregado está vazio."
        )

    logger.info(
        "✅ Snapshot carregado | "
        f"rows={len(df):,} | "
        f"columns={len(df.columns)}"
    )

    return df


@task(
    name="Auditar Snapshot Gold",
)
def audit_gold_snapshot_task(
    df: pd.DataFrame,
    *,
    dataset_role: str,
):
    """
    Executa o Data Quality Gate do snapshot antes
    que os dados avancem para otimização/treinamento.
    """

    report = audit_gold_snapshot(
        df
    )

    logger.info(
        "✅ Auditoria Gold aprovada | "
        f"role={dataset_role} | "
        f"rows={report.rows:,} | "
        f"range={report.start_date} → "
        f"{report.end_date} | "
        "missing_timestamps="
        f"{report.missing_timestamps} | "
        f"gaps={report.gap_count} | "
        f"features={report.feature_count}"
    )

    return report


@task(
    name="Dividir Snapshot em Treino e OOT",
)
def split_expanding_window_task(
    df: pd.DataFrame,
    *,
    oot_year: int,
    oot_month: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Wrapper Prefect da função pura de split temporal.
    """

    df_train, df_test = (
        split_expanding_window_snapshot(
            df,
            oot_year=oot_year,
            oot_month=oot_month,
        )
    )

    oot_start, oot_end = (
        get_local_month_bounds_utc(
            oot_year,
            oot_month,
        )
    )

    logger.info(
        "✂️ Split temporal concluído | "
        f"train_rows={len(df_train):,} | "
        f"oot_rows={len(df_test):,} | "
        f"oot_utc=[{oot_start}, {oot_end})"
    )

    return df_train, df_test


def build_training_dataset_labels(
    df_train: pd.DataFrame,
    *,
    oot_year: int,
    oot_month: int,
) -> tuple[str, str]:
    """
    Constrói nomes determinísticos para os artefatos
    de treino e OOT derivados do snapshot auditado.
    """

    if df_train.empty:
        raise ValueError(
            "Não é possível gerar labels "
            "a partir de treino vazio."
        )

    train_start = pd.to_datetime(
        df_train["date"],
        errors="raise",
    ).min()

    train_start_str = (
        train_start.strftime(
            "%Y_%m"
        )
    )

    cutoff_str = (
        f"{oot_year}_"
        f"{oot_month:02d}"
    )

    train_file_label = (
        f"train_wind_energy_"
        f"{train_start_str}"
        f"_expanding_up_to_"
        f"{cutoff_str}.parquet"
    )

    test_file_label = (
        "oot_test_wind_energy_"
        f"{cutoff_str}.parquet"
    )

    return (
        train_file_label,
        test_file_label,
    )


@task(
    name="Persistir Datasets de Treino e OOT",
)
def persist_training_datasets_task(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    oot_year: int,
    oot_month: int,
) -> tuple[str, str]:
    """
    Persiste treino e OOT somente depois que o
    snapshot consolidado foi auditado e dividido
    pelo contrato temporal oficial.
    """

    (
        train_file_label,
        test_file_label,
    ) = build_training_dataset_labels(
        df_train,
        oot_year=oot_year,
        oot_month=oot_month,
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
        "✅ Datasets de treino e OOT "
        "persistidos com sucesso | "
        f"train={train_file_label} | "
        f"oot={test_file_label}"
    )

    return (
        train_file_label,
        test_file_label,
    )


# ==============================================================================
# PASSO 2: OTIMIZAÇÃO E TREINAMENTO AGNÓSTICO
# ==============================================================================


@task(
    name="Otimizar Hiperparâmetros (Agnóstico)",
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
    name="Treinar Modelo (Agnóstico)",
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
    name="Quality Gate (Champion vs Challenger)",
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
    snapshot_path: str = (
        DEFAULT_TRAINING_GOLD_SNAPSHOT_PATH
    ),
    oot_year: int = DEFAULT_OOT_YEAR,
    oot_month: int = DEFAULT_OOT_MONTH,
):
    """
    Orquestrador agnóstico de treinamento contínuo.

    O trainer é obrigatório. A etapa de otimização é opcional:

    - um optimizer registrado executa antes do treinamento;
    - ``optimizer_name="none"`` pula explicitamente essa etapa;
    - nomes desconhecidos falham antes da execução do pipeline.

    Quando não há otimização, o trainer recebe ``None`` em
    ``best_params`` e ``optimization_run_id``.

    O dataset de origem é sempre um snapshot Gold explícito.
    Antes do split, o snapshot passa pelo Data Quality Gate.
    O mês OOT é interpretado em ``America/Sao_Paulo`` e
    convertido para os limites UTC usados pelo dataset.
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

    df_gold = load_gold_snapshot_task(
        snapshot_path=snapshot_path,
    )

    audit_gold_snapshot_task(
        df_gold,
        dataset_role="consolidated",
    )

    (
        df_train,
        df_test,
    ) = split_expanding_window_task(
        df_gold,
        oot_year=oot_year,
        oot_month=oot_month,
    )

    # Defesa em profundidade: o dataset consolidado já
    # foi auditado, mas validamos também os artefatos que
    # serão efetivamente consumidos pelo modelo.
    audit_gold_snapshot_task(
        df_train,
        dataset_role="train",
    )

    audit_gold_snapshot_task(
        df_test,
        dataset_role="oot",
    )

    (
        train_file,
        test_file,
    ) = persist_training_datasets_task(
        df_train,
        df_test,
        oot_year=oot_year,
        oot_month=oot_month,
    )

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
