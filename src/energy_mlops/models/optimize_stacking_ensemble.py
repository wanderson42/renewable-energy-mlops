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
from energy_mlops.models.interfaces import OptimizationResult
from energy_mlops.models.temporal_stacking import TemporalStackingRegressor

'''
                       target_fc
                           │
                           ▼
                    treinamento
                           │
                           ▼
                    predicted_fc
                           │
                           │ × capacidade_mw
                           ▼
                    predicted_mw
                           │
              ┌────────────┴────────────┐
              │                         │
       wind_generation_mw        capacidade máxima
              │                         │
              └────── MAE MW ───────────┘
                           │
                           ▼
                       nMAE (%)
                           │
                           ▼
                     Optuna minimize
'''
def objective(
    trial,
    X,
    y_fc,
    y_mw,
    capacity,
    n_splits,
    stacking_n_splits=5,
):
    """
    Função objetivo do Optuna para o Temporal Stacking Ensemble.

    O modelo é treinado sobre target_fc, mas a função objetivo
    é avaliada usando a métrica operacional nMAE (%), calculada
    em MW.

    Estrutura temporal:

        Outer TimeSeriesSplit
                │
                ├── X_train
                │      │
                │      └── TemporalStackingRegressor
                │              │
                │              └── Inner TimeSeriesSplit
                │
                └── X_validation futuro
                           │
                           └── nMAE (%)

    Assim:
    - o treinamento permanece em fator de capacidade;
    - a seleção de hiperparâmetros é feita pela métrica
      operacional utilizada no restante do pipeline;
    - nenhum fold utiliza informações futuras.
    """

    # ==========================================================
    # LightGBM
    # ==========================================================
    lgb_params = {
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
        "objective": "regression",
        "metric": "mae",
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    # ==========================================================
    # XGBoost
    # ==========================================================
    xgb_params = {
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
        "objective": "reg:absoluteerror",
        "random_state": 42,
        "n_jobs": -1,
    }

    # ==========================================================
    # Random Forest
    # ==========================================================
    rf_params = {
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
        "random_state": 42,
        "n_jobs": -1,
    }

    # ==========================================================
    # Cross-validation temporal externa
    #
    # Mede a generalização do trial em períodos futuros.
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
        # Target usado para treinamento
        # ------------------------------------------------------
        y_tr_fc = y_fc.iloc[train_idx]
        y_val_fc = y_fc.iloc[val_idx]

        # ------------------------------------------------------
        # Ground truth operacional + capacidade
        # ------------------------------------------------------
        y_val_mw = y_mw.iloc[val_idx]
        cap_val = capacity.iloc[val_idx]

        # ------------------------------------------------------
        # Cross-validation temporal interna
        #
        # O TemporalStackingRegressor precisa produzir
        # previsões OOF causais para treinar o meta-learner.
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
        # Modelos base
        # ------------------------------------------------------
        lgb_model = lgb.LGBMRegressor(
            **lgb_params
        )

        xgb_model = XGBRegressor(
            **xgb_params
        )

        rf_model = RandomForestRegressor(
            **rf_params
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
            estimators=[
                ("lgbm", lgb_model),
                ("xgboost", xgb_model),
                ("rf", rf_model),
            ],
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
    # Métricas auxiliares armazenadas no Trial
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

    # ==========================================================
    # O Optuna minimiza nMAE (%)
    # ==========================================================
    return mean_nmae


def run_optimization(
    df_train: pd.DataFrame,
    train_file: str,
    n_trials: int = 20,
    n_splits: int = 3,
    stacking_n_splits: int = 5,
) -> OptimizationResult:
    """
    Executa a otimização temporal do Stacking Ensemble.

    O modelo é treinado em target_fc, mas a busca de
    hiperparâmetros minimiza o nMAE (%) calculado em MW.

    Retorna:
        {
            "optimization_run_id": str,
            "best_params": {
                "lgb_params": {...},
                "xgb_params": {...},
                "rf_params": {...},
            },
        }
    """

    # ==========================================================
    # Contrato das features
    # ==========================================================
    X_train = select_model_features(
        df_train
    )

    # Target usado pelo modelo
    y_train_fc = df_train[
        "target_fc"
    ]

    # Ground truth operacional
    y_train_mw = df_train[
        "wind_generation_mw"
    ]

    # Capacidade usada para converter FC -> MW
    # e normalizar o erro.
    capacity_train = df_train[
        "capacidade_mw"
    ]

    print(
        "🚀 Iniciando otimização do Temporal Stacking "
        f"via Optuna com {n_trials} trials, "
        f"{n_splits} outer CV splits e "
        f"{stacking_n_splits} inner CV splits..."
    )

    # ==========================================================
    # Estudo Optuna
    # ==========================================================
    study = optuna.create_study(
        direction="minimize",
        study_name="Ensemble_Optimization",
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train_fc,
            y_train_mw,
            capacity_train,
            n_splits=n_splits,
            stacking_n_splits=stacking_n_splits,
        ),
        n_trials=n_trials,
    )

    # ==========================================================
    # Melhor Trial
    # ==========================================================
    best_trial = study.best_trial
    best_params_raw = best_trial.params

    # ==========================================================
    # Reconstrução dos parâmetros por modelo
    # ==========================================================
    final_params = {
        "lgb_params": {
            key.replace("lgb_", ""): value
            for key, value in best_params_raw.items()
            if key.startswith("lgb_")
        },
        "xgb_params": {
            key.replace("xgb_", ""): value
            for key, value in best_params_raw.items()
            if key.startswith("xgb_")
        },
        "rf_params": {
            key.replace("rf_", ""): value
            for key, value in best_params_raw.items()
            if key.startswith("rf_")
        },
    }

    # ==========================================================
    # MLflow
    # ==========================================================
    mlflow.set_tracking_uri(
        settings.MLFLOW_TRACKING_URI
    )

    experiment_name = (
        "wind_power_optimization_bahia"
    )

    mlflow.set_experiment(
        experiment_name
    )

    with mlflow.start_run(
        run_name="Optuna_Best_Trial"
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
        # Hiperparâmetros vencedores
        # ------------------------------------------------------
        mlflow.log_params(
            best_params_raw
        )

        # ------------------------------------------------------
        # Governança do processo de otimização
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
        # Artefato de parâmetros para o trainer
        # ------------------------------------------------------
        mlflow.log_dict(
            final_params,
            "best_ensemble_params.json",
        )

    print(
        "💾 Hiperparâmetros otimizados por nMAE "
        "salvos com sucesso no MLflow!"
    )

    return {
        "optimization_run_id": (
            optimization_run_id
        ),
        "best_params": final_params,
    }