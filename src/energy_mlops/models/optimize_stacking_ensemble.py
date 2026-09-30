import lightgbm as lgb
import mlflow
import mlflow.data
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from energy_mlops.config import settings
from energy_mlops.data.feature_utils import select_model_features
from energy_mlops.models.ensemble_config import (
    DEFAULT_BASE_ESTIMATORS,
    ESTIMATOR_LABEL_BY_NAME,
    PARAM_GROUP_BY_ESTIMATOR,
    PARAM_PREFIX_BY_ESTIMATOR,
    validate_enabled_estimators,
)
from energy_mlops.models.interfaces import OptimizationResult
from energy_mlops.models.temporal_stacking import (
    TemporalStackingRegressor,
)

OPTIMIZATION_EXPERIMENT_NAME = (
    "wind_power_optimization_bahia"
)


def _get_fixed_estimator_params(
    estimator_name: str,
) -> dict:
    """
    Retorna os hiperparâmetros fixos usados tanto durante a
    otimização quanto no treinamento final.

    Isso garante que o modelo avaliado pelo Optuna e o modelo
    posteriormente treinado utilizem o mesmo contrato.
    """

    if estimator_name == "lgbm":
        return {
            "objective": "regression",
            "metric": "mae",
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }

    if estimator_name == "xgboost":
        return {
            "objective": "reg:absoluteerror",
            "random_state": 42,
            "n_jobs": -1,
        }

    if estimator_name == "rf":
        return {
            "random_state": 42,
            "n_jobs": -1,
        }

    raise RuntimeError(
        "Estimador inesperado após validação: "
        f"{estimator_name}"
    )


def _suggest_estimator_params(
    trial,
    enabled_estimators: tuple[str, ...],
) -> dict[str, dict]:
    """
    Constrói somente os espaços de busca dos estimadores
    habilitados.

    Portanto, por exemplo, uma arquitetura LGBM + RF não cria
    nenhuma dimensão XGBoost no espaço de busca do Optuna.
    """

    params_by_estimator: dict[str, dict] = {}

    if "lgbm" in enabled_estimators:
        tuned_params = {
            "n_estimators": trial.suggest_int(
                "lgb_n_estimators",
                100,
                500,
            ),
            "learning_rate": trial.suggest_float(
                "lgb_learning_rate",
                0.01,
                0.1,
                log=True,
            ),
            "num_leaves": trial.suggest_int(
                "lgb_num_leaves",
                20,
                100,
            ),
            "max_depth": trial.suggest_int(
                "lgb_max_depth",
                3,
                10,
            ),
        }

        params_by_estimator["lgbm"] = {
            **tuned_params,
            **_get_fixed_estimator_params("lgbm"),
        }

    if "xgboost" in enabled_estimators:
        tuned_params = {
            "n_estimators": trial.suggest_int(
                "xgb_n_estimators",
                100,
                500,
            ),
            "learning_rate": trial.suggest_float(
                "xgb_learning_rate",
                0.01,
                0.1,
                log=True,
            ),
            "max_depth": trial.suggest_int(
                "xgb_max_depth",
                3,
                10,
            ),
            "subsample": trial.suggest_float(
                "xgb_subsample",
                0.6,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "xgb_colsample_bytree",
                0.6,
                1.0,
            ),
        }

        params_by_estimator["xgboost"] = {
            **tuned_params,
            **_get_fixed_estimator_params("xgboost"),
        }

    if "rf" in enabled_estimators:
        tuned_params = {
            "n_estimators": trial.suggest_int(
                "rf_n_estimators",
                100,
                300,
            ),
            "max_depth": trial.suggest_int(
                "rf_max_depth",
                5,
                20,
            ),
            "min_samples_split": trial.suggest_int(
                "rf_min_samples_split",
                2,
                20,
            ),
        }

        params_by_estimator["rf"] = {
            **tuned_params,
            **_get_fixed_estimator_params("rf"),
        }

    return params_by_estimator


def _build_estimators(
    params_by_estimator: dict[str, dict],
    enabled_estimators: tuple[str, ...],
) -> list[tuple[str, object]]:
    """
    Instancia os modelos base respeitando exatamente a ordem
    definida em enabled_estimators.
    """

    estimators: list[tuple[str, object]] = []

    for name in enabled_estimators:
        params = params_by_estimator[name]

        if name == "lgbm":
            estimator = lgb.LGBMRegressor(
                **params
            )

        elif name == "xgboost":
            estimator = XGBRegressor(
                **params
            )

        elif name == "rf":
            estimator = RandomForestRegressor(
                **params
            )

        else:
            raise RuntimeError(
                "Estimador inesperado após validação: "
                f"{name}"
            )

        estimators.append(
            (
                name,
                estimator,
            )
        )

    return estimators


def _reconstruct_best_params(
    best_params_raw: dict,
    enabled_estimators: tuple[str, ...],
) -> dict[str, dict]:
    """
    Reconstrói os parâmetros vencedores no contrato esperado
    pelo trainer.

    Além dos parâmetros sugeridos pelo Optuna, preserva também
    os parâmetros fixos usados durante a otimização.
    """

    final_params: dict[str, dict] = {}

    for estimator_name in enabled_estimators:
        param_group = PARAM_GROUP_BY_ESTIMATOR[
            estimator_name
        ]

        prefix = (
            PARAM_PREFIX_BY_ESTIMATOR[
                estimator_name
            ]
            + "_"
        )

        tuned_params = {
            key.removeprefix(prefix): value
            for key, value in best_params_raw.items()
            if key.startswith(prefix)
        }

        final_params[param_group] = {
            **tuned_params,
            **_get_fixed_estimator_params(
                estimator_name
            ),
        }

    return final_params


def objective(
    trial,
    X,
    y_fc,
    y_mw,
    capacity,
    n_splits,
    stacking_n_splits=5,
    enabled_estimators: tuple[str, ...] = (
        DEFAULT_BASE_ESTIMATORS
    ),
):
    """
    Função objetivo do Optuna para o Temporal Stacking.

    O modelo é treinado em target_fc, enquanto a seleção dos
    hiperparâmetros minimiza o nMAE (%) calculado em MW.

    A validação possui dois níveis temporais:

        Outer TimeSeriesSplit
                │
                ├── treino passado
                │       │
                │       └── TemporalStackingRegressor
                │               │
                │               └── Inner TimeSeriesSplit
                │
                └── validação futura

    Nenhum fold utiliza informação futura.
    """

    enabled_estimators = (
        validate_enabled_estimators(
            enabled_estimators
        )
    )

    # ==========================================================
    # Espaço de busca específico da arquitetura
    # ==========================================================

    params_by_estimator = (
        _suggest_estimator_params(
            trial,
            enabled_estimators,
        )
    )

    # ==========================================================
    # Cross-validation temporal externa
    # ==========================================================

    outer_cv = TimeSeriesSplit(
        n_splits=n_splits,
    )

    fold_nmaes = []
    fold_mae_fc_pcts = []
    fold_r2s = []

    for train_idx, val_idx in outer_cv.split(X):
        # ------------------------------------------------------
        # Features
        # ------------------------------------------------------

        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        # ------------------------------------------------------
        # Target do modelo
        # ------------------------------------------------------

        y_tr_fc = y_fc.iloc[train_idx]
        y_val_fc = y_fc.iloc[val_idx]

        # ------------------------------------------------------
        # Ground truth operacional
        # ------------------------------------------------------

        y_val_mw = y_mw.iloc[val_idx]
        cap_val = capacity.iloc[val_idx]

        # ------------------------------------------------------
        # Cross-validation temporal interna
        # ------------------------------------------------------

        inner_n_splits = min(
            stacking_n_splits,
            len(X_tr) - 1,
        )

        if inner_n_splits < 2:
            raise ValueError(
                "Número insuficiente de amostras para "
                "TemporalStackingRegressor dentro do "
                "cross-validation do Optuna."
            )

        inner_cv = TimeSeriesSplit(
            n_splits=inner_n_splits,
        )

        # ------------------------------------------------------
        # Modelos base da arquitetura selecionada
        # ------------------------------------------------------

        estimators = _build_estimators(
            params_by_estimator,
            enabled_estimators,
        )

        # ------------------------------------------------------
        # Meta-learner
        # ------------------------------------------------------

        meta_learner = LinearRegression(
            positive=True,
            fit_intercept=False,
        )

        # ------------------------------------------------------
        # Temporal Stacking
        # ------------------------------------------------------

        ensemble = TemporalStackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=inner_cv,
        )

        ensemble.fit(
            X_tr,
            y_tr_fc,
        )

        # ======================================================
        # Predição em fator de capacidade
        # ======================================================

        predictions_fc = ensemble.predict(
            X_val
        )

        predictions_fc = np.asarray(
            predictions_fc,
            dtype=float,
        )

        # ======================================================
        # Conversão FC -> MW
        # ======================================================

        capacity_val = cap_val.to_numpy(
            dtype=float
        )

        actual_mw = y_val_mw.to_numpy(
            dtype=float
        )

        predictions_mw = (
            predictions_fc
            * capacity_val
        )

        # ======================================================
        # Métrica primária: nMAE (%)
        # ======================================================

        fold_mae_mw = mean_absolute_error(
            actual_mw,
            predictions_mw,
        )

        max_capacity = float(
            np.max(capacity_val)
        )

        if max_capacity <= 0:
            raise ValueError(
                "A capacidade máxima do fold deve ser "
                "maior que zero para calcular o nMAE."
            )

        fold_nmae_pct = (
            fold_mae_mw
            / max_capacity
        ) * 100

        # ======================================================
        # Métricas diagnósticas
        # ======================================================

        fold_mae_fc_pct = (
            mean_absolute_error(
                y_val_fc,
                predictions_fc,
            )
            * 100
        )

        fold_r2 = r2_score(
            y_val_fc,
            predictions_fc,
        )

        fold_nmaes.append(
            fold_nmae_pct
        )

        fold_mae_fc_pcts.append(
            fold_mae_fc_pct
        )

        fold_r2s.append(
            fold_r2
        )

    # ==========================================================
    # Agregação dos folds
    # ==========================================================

    mean_nmae = float(
        np.mean(fold_nmaes)
    )

    std_nmae = float(
        np.std(fold_nmaes)
    )

    mean_mae_fc_pct = float(
        np.mean(fold_mae_fc_pcts)
    )

    mean_r2 = float(
        np.mean(fold_r2s)
    )

    # ==========================================================
    # Métricas auxiliares do Trial
    # ==========================================================

    trial.set_user_attr(
        "cv_nmae_std",
        std_nmae,
    )

    trial.set_user_attr(
        "cv_mae_fc_pct_mean",
        mean_mae_fc_pct,
    )

    trial.set_user_attr(
        "cv_r2_mean",
        mean_r2,
    )

    return mean_nmae


def run_optimization(
    df_train: pd.DataFrame,
    train_file: str,
    n_trials: int = 20,
    n_splits: int = 3,
    stacking_n_splits: int = 5,
    enabled_estimators: tuple[str, ...] = (
        DEFAULT_BASE_ESTIMATORS
    ),
) -> OptimizationResult:
    """
    Executa a otimização temporal do Temporal Stacking.

    O espaço de busca é condicionado à arquitetura definida
    em enabled_estimators.

    Exemplos:

        ("lgbm", "xgboost", "rf")
        ("lgbm", "rf")

    Retorna a Optimization Run e os parâmetros completos
    necessários para reproduzir o modelo vencedor.
    """

    enabled_estimators = (
        validate_enabled_estimators(
            enabled_estimators
        )
    )

    architecture_label = "_".join(
        ESTIMATOR_LABEL_BY_NAME[name]
        for name in enabled_estimators
    )

    architecture_str = ",".join(
        enabled_estimators
    )

    # ==========================================================
    # Contrato das features
    # ==========================================================

    X_train = select_model_features(
        df_train
    )

    y_train_fc = df_train[
        "target_fc"
    ]

    y_train_mw = df_train[
        "wind_generation_mw"
    ]

    capacity_train = df_train[
        "capacidade_mw"
    ]

    print(
        "🚀 Iniciando otimização do Temporal Stacking "
        f"[{architecture_str}] via Optuna com "
        f"{n_trials} trials, "
        f"{n_splits} outer CV splits e "
        f"{stacking_n_splits} inner CV splits..."
    )

    # ==========================================================
    # Estudo Optuna
    # ==========================================================

    study = optuna.create_study(
        direction="minimize",
        study_name=(
            f"Ensemble_Optimization_"
            f"{architecture_label}"
        ),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train_fc,
            y_train_mw,
            capacity_train,
            n_splits=n_splits,
            stacking_n_splits=(
                stacking_n_splits
            ),
            enabled_estimators=(
                enabled_estimators
            ),
        ),
        n_trials=n_trials,
    )

    # ==========================================================
    # Melhor Trial
    # ==========================================================

    best_trial = study.best_trial

    best_params_raw = dict(
        best_trial.params
    )

    final_params = _reconstruct_best_params(
        best_params_raw,
        enabled_estimators,
    )

    # ==========================================================
    # MLflow
    # ==========================================================

    mlflow.set_tracking_uri(
        settings.MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        OPTIMIZATION_EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name=(
            f"Optuna_Best_Trial_"
            f"{architecture_label}"
        )
    ) as run:
        optimization_run_id = (
            run.info.run_id
        )

        # ------------------------------------------------------
        # Data Lineage
        # ------------------------------------------------------

        mlflow.log_input(
            mlflow.data.from_pandas(
                df=X_train,
                source=(
                    f"s3://{settings.RUSTFS_BUCKET}"
                    f"/gold/{train_file}"
                ),
                name=train_file.replace(
                    ".parquet",
                    "",
                ),
            ),
            context="training",
        )

        # ------------------------------------------------------
        # Parâmetros sugeridos pelo Optuna
        # ------------------------------------------------------

        mlflow.log_params(
            best_params_raw
        )

        # ------------------------------------------------------
        # Arquitetura
        # ------------------------------------------------------

        mlflow.log_param(
            "base_estimators",
            architecture_str,
        )

        mlflow.log_param(
            "n_base_estimators",
            len(enabled_estimators),
        )

        mlflow.set_tag(
            "ensemble_architecture",
            architecture_str,
        )

        # ------------------------------------------------------
        # Governança da otimização
        # ------------------------------------------------------

        mlflow.log_param(
            "outer_cv_n_splits",
            n_splits,
        )

        mlflow.log_param(
            "stacking_cv_n_splits",
            stacking_n_splits,
        )

        mlflow.log_param(
            "num_features",
            len(X_train.columns),
        )

        mlflow.log_param(
            "optimization_metric",
            "nmae_pct",
        )

        mlflow.log_param(
            "training_samples",
            len(df_train),
        )

        # ------------------------------------------------------
        # Janela temporal
        # ------------------------------------------------------

        if "date" in df_train.columns:
            train_dates = pd.to_datetime(
                df_train["date"]
            )

            mlflow.log_param(
                "training_window_start",
                train_dates.min().isoformat(),
            )

            mlflow.log_param(
                "training_window_end",
                train_dates.max().isoformat(),
            )

        # ------------------------------------------------------
        # Métrica primária
        # ------------------------------------------------------

        mlflow.log_metric(
            "best_cv_nmae_pct_mean",
            best_trial.value,
        )

        mlflow.log_metric(
            "best_cv_nmae_pct_std",
            best_trial.user_attrs.get(
                "cv_nmae_std",
                0.0,
            ),
        )

        # ------------------------------------------------------
        # Métricas diagnósticas
        # ------------------------------------------------------

        mlflow.log_metric(
            "best_cv_mae_fc_pct_mean",
            best_trial.user_attrs.get(
                "cv_mae_fc_pct_mean",
                0.0,
            ),
        )

        mlflow.log_metric(
            "best_cv_r2_mean",
            best_trial.user_attrs.get(
                "cv_r2_mean",
                0.0,
            ),
        )

        # ------------------------------------------------------
        # Contrato completo usado pelo trainer
        # ------------------------------------------------------

        mlflow.log_dict(
            final_params,
            "best_ensemble_params.json",
        )

    print(
        "💾 Hiperparâmetros do Temporal Stacking "
        f"[{architecture_str}] otimizados por nMAE "
        "e salvos com sucesso no MLflow!"
    )

    return {
        "optimization_run_id": (
            optimization_run_id
        ),
        "best_params": final_params,
    }