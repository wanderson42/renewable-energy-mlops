import json

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from energy_mlops.config import settings
from energy_mlops.data.feature_utils import select_model_features
from energy_mlops.models.ensemble_config import (
    DEFAULT_BASE_ESTIMATORS,
    PARAM_GROUP_BY_ESTIMATOR,
    PARAM_PREFIX_BY_ESTIMATOR,
    validate_enabled_estimators,
)
from energy_mlops.models.temporal_stacking import (
    TemporalStackingRegressor,
)

'''
TimeSeriesSplit

Fold 1
passado ─────────► futuro
TRAIN              VALID
  │                  │
  └── base models ───┘
           │
       OOF predictions

Fold 2
passado ─────────────────► futuro
TRAIN                      VALID
  │                          │
  └────── base models ───────┘
               │
          OOF predictions

               ↓

        todas as OOF válidas
               ↓
    LinearRegression
        meta-learner
               ↓

base models são refitados em TODO o X_train
               ↓
         modelo final
'''


ESTIMATOR_NAME_TOKENS = {
    "lgbm": "lgb",
    "xgboost": "xgb",
    "rf": "rf",
}


def build_architecture_names(
    enabled_estimators: tuple[str, ...],
) -> tuple[str, str]:
    """
    Gera nomes determinísticos da Run e do
    Registered Model a partir da arquitetura.
    """

    unknown_estimators = (
        set(enabled_estimators)
        - set(ESTIMATOR_NAME_TOKENS)
    )

    if unknown_estimators:
        raise ValueError(
            "Estimators sem nomenclatura definida: "
            f"{sorted(unknown_estimators)}"
        )

    architecture_slug = "_".join(
        ESTIMATOR_NAME_TOKENS[name]
        for name in enabled_estimators
    )

    run_name = (
        f"Stacking_"
        f"{architecture_slug.upper()}"
        f"_Bahia"
    )

    model_name = (
        f"ensemble_"
        f"{architecture_slug}"
        f"_bahia"
    )

    return run_name, model_name


def prepare_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple:
    """Separa features e targets usando o contrato ModelFeatureSchema."""

    X_train = select_model_features(df_train)
    X_test = select_model_features(df_test)

    return (
        X_train,
        X_test,
        df_train["target_fc"],
        df_test["target_fc"],
        df_train["wind_generation_mw"],
        df_test["wind_generation_mw"],
        df_train["capacidade_mw"],
        df_test["capacidade_mw"],
    )


def train_stacking_regressor(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    train_file: str,
    test_file: str,
    best_params: dict | None = None,
    optimization_run_id: str | None = None,
    enabled_estimators: tuple[str, ...] = DEFAULT_BASE_ESTIMATORS,
) -> tuple[str, float]:
    """
    Treina o modelo, registra todos os metadados (SHAP, trusted_types, métricas ricas) 
    e retorna o ID da Run e o MAE para o orquestrador (Prefect).
    """

    if best_params is None:
        raise ValueError(
            "O Temporal Stacking requer 'best_params' "
            "produzidos por uma etapa de otimização."
        )

    if optimization_run_id is None:
        raise ValueError(
            "O Temporal Stacking requer "
            "'optimization_run_id' para rastreabilidade."
        )

    enabled_estimators = validate_enabled_estimators(enabled_estimators)

    run_name, model_name = (
        build_architecture_names(
            enabled_estimators
        )
    )

    X_train, X_test, y_train_fc, y_test_fc, y_train_mw, y_test_mw, cap_train, cap_test = (
        prepare_features(df_train, df_test)
    )

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    experiment_name = "wind_power_forecasting_bahia"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:

        architecture_str = ",".join(
            enabled_estimators
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

        # Temporário — benchmark de simplificação v0.2.0
        mlflow.set_tag(
            "experiment_family",
            "ensemble_simplification_v0.2.0",
        )

        mlflow.set_tag(
            "architecture_run_name",
            run_name,
        )

        mlflow.set_tag(
            "registered_model_name",
            model_name,
        )

        # ------------------------------------------------------
        # Linhagem da otimização
        # ------------------------------------------------------

        mlflow.set_tag(
            "optimization_run_id",
            optimization_run_id,
        )

        mlflow.set_tag(
            "optimization_experiment",
            "wind_power_optimization_bahia",
        )

        run_id = run.info.run_id
        
        mlflow.log_input(
            mlflow.data.from_pandas(
                df=X_train,
                source=f"s3://{settings.RUSTFS_BUCKET}/gold/{train_file}",
                name=train_file.replace(".parquet", "")),
                context="training")

        mlflow.log_input(
            mlflow.data.from_pandas(
                df=X_test,
                source=f"s3://{settings.RUSTFS_BUCKET}/gold/{test_file}",
                name=test_file.replace(".parquet", "")),
                context="testing")

        num_features_challenger = len(X_train.columns)
        mlflow.log_param("num_features", num_features_challenger)

        mlflow.log_dict(
            best_params,
            "optimization_params_used.json",
        )

        # ==========================================================
        # 1. Parâmetros produzidos pela Optimization Run
        # ==========================================================

        params_by_estimator = {
            "lgbm": best_params.get(
                "lgb_params",
                {},
            ),
            "xgboost": best_params.get(
                "xgb_params",
                {},
            ),
            "rf": best_params.get(
                "rf_params",
                {},
            ),
        }

        # Exige parâmetros apenas dos estimadores habilitados.
        missing_groups = [
            PARAM_GROUP_BY_ESTIMATOR[name]
            for name in enabled_estimators
            if not params_by_estimator[name]
        ]

        if missing_groups:
            raise ValueError(
                "Parâmetros de otimização ausentes: "
                f"{missing_groups}. "
                f"Optimization Run: {optimization_run_id}"
            )


        # ==========================================================
        # 2. Logging dos hiperparâmetros utilizados
        # ==========================================================

        for name in enabled_estimators:
            prefix = PARAM_PREFIX_BY_ESTIMATOR[name]

            mlflow.log_params(
                {
                    f"{prefix}_{key}": value
                    for key, value
                    in params_by_estimator[name].items()
                }
            )


        # ==========================================================
        # 3. Construção dinâmica dos modelos base
        # ==========================================================

        estimators = []

        for name in enabled_estimators:
            if name == "lgbm":
                estimator = lgb.LGBMRegressor(
                    **params_by_estimator[name]
                )

            elif name == "xgboost":
                estimator = XGBRegressor(
                    **params_by_estimator[name]
                )

            elif name == "rf":
                estimator = RandomForestRegressor(
                    **params_by_estimator[name]
                )

            else:
                # Este caso já deve ter sido bloqueado por
                # validate_enabled_estimators().
                raise RuntimeError(
                    f"Estimador inesperado após validação: {name}"
                )

            estimators.append(
                (
                    name,
                    estimator,
                )
            )


        # ==========================================================
        # 4. Meta-learner + validação temporal
        # ==========================================================

        meta_learner = LinearRegression(
            positive=True,
            fit_intercept=False,
        )

        cv_strategy = TimeSeriesSplit(
            n_splits=5,
        )


        # ==========================================================
        # 5. Temporal Stacking
        # ==========================================================

        ensemble = TemporalStackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=cv_strategy,
        )

        print(
            "🚀 A treinar Stacking Ensemble "
            f"({X_train.shape[0]} amostras, "
            f"{num_features_challenger} features, "
            f"estimators={list(enabled_estimators)})..."
        )

        ensemble.fit(
            X_train,
            y_train_fc,
        )


        # ==========================================================
        # 6. Pesos OOF aprendidos pelo meta-learner
        # ==========================================================

        estimator_names = [
            name
            for name, _ in estimators
        ]

        meta_coefs = (
            ensemble.final_estimator_.coef_
        )

        meta_coef_by_estimator = dict(
            zip(
                estimator_names,
                meta_coefs,
                strict=True,
            )
        )

        print(
            "🧠 Pesos OOF aprendidos pelo Meta-Learner "
            f"({', '.join(estimator_names)}): "
            f"{meta_coefs}"
        )

        mlflow.log_params(
            {
                f"meta_coef_{name}": float(coef)
                for name, coef
                in meta_coef_by_estimator.items()
            }
        )

        mlflow.log_param(
            "stacking_cv_strategy",
            type(cv_strategy).__name__,
        )

        mlflow.log_param(
            "stacking_cv_n_splits",
            cv_strategy.n_splits,
        )


        # Previsões
        train_preds_fc = ensemble.predict(X_train)
        train_preds_mw = train_preds_fc * cap_train
        train_mae_mw = mean_absolute_error(y_train_mw, train_preds_mw)
        train_nmae_pct = (train_mae_mw / cap_train.max()) * 100
        train_mae_fc_pct = mean_absolute_error(y_train_fc, train_preds_fc) * 100

        test_preds_fc = ensemble.predict(X_test)
        test_preds_mw = test_preds_fc * cap_test
        test_mae_mw = mean_absolute_error(y_test_mw, test_preds_mw)
        test_nmae_pct = (test_mae_mw / cap_test.max()) * 100
        test_mae_fc_pct = mean_absolute_error(y_test_fc, test_preds_fc) * 100
        test_r2 = r2_score(y_test_fc, test_preds_fc)

        mlflow.log_metric("train_mae_mw", train_mae_mw)
        mlflow.log_metric("train_nmae_pct", train_nmae_pct)
        mlflow.log_metric("train_mae_fc_pct", train_mae_fc_pct)
        # Mantendo os nomes exatos originais
        mlflow.log_metric("oot_mae_mw", test_mae_mw)
        mlflow.log_metric("oot_nmae_pct", test_nmae_pct)
        mlflow.log_metric("oot_mae_fc_pct", test_mae_fc_pct)
        mlflow.log_metric("oot_r2_score", test_r2)


        print("✅ Treino do Stacking Ensemble concluído!")
        print(f"📊 Desempenho TREINO: MAE: {train_mae_mw:.2f} MW | nMAE: {train_nmae_pct:.2f}% | MAE_FC: {train_mae_fc_pct:.2f}%")
        print(f"🔥 Desempenho OOT:    MAE: {test_mae_mw:.2f} MW | nMAE: {test_nmae_pct:.2f}% | MAE_FC: {test_mae_fc_pct:.2f}% | R2: {test_r2:.4f}")

        # ==============================================================================
        # 📑 REGISTRO DINÂMICO DA FICHA TÉCNICA E METADADOS DO MODELO
        # ==============================================================================
        base_estimators_info = [
            {"name": name, "type": type(est).__name__}
            for name, est in ensemble.named_estimators_.items()
        ]

        meta_weights = {
            name: round(
                float(coef),
                4,
            )
            for name, coef
            in meta_coef_by_estimator.items()
        }

        model_summary = {
            "model_name": "ensemble_lgb_xgb_rf_bahia",
            "architecture": type(ensemble).__name__,
            "cv_strategy": type(cv_strategy).__name__,
            "meta_learner": type(ensemble.final_estimator_).__name__,
            "base_estimators": base_estimators_info,
            "meta_weights": meta_weights,
            "num_features": num_features_challenger,
            "metrics": {
                "r2_score": round(float(test_r2), 4),
                "oot_mae_mw": round(float(test_mae_mw), 2),
                "oot_nmae_pct": round(float(test_nmae_pct), 2),
                "oot_mae_fc_pct": round(float(test_mae_fc_pct), 2),
            }
        }

        mlflow.log_dict(model_summary, "model_summary.json")
        
        architecture_summary = (
            f"{type(ensemble).__name__} "
            f"({', '.join(meta_weights.keys())}) "
            f"-> "
            f"{type(ensemble.final_estimator_).__name__}"
        )

        mlflow.set_tag(
            "architecture_str",
            architecture_summary,
        )

        mlflow.set_tag(
            "meta_weights_json",
            json.dumps(meta_weights),
        )

        # ==============================================================================
        # 🔍 EXPLICABILIDADE SHAP (Restaurada)
        # ==============================================================================
        print("🔍 Gerando SHAP Values e Gráficos Explicativos...")
        lgbm_instance = ensemble.named_estimators_["lgbm"]
        explainer = shap.TreeExplainer(lgbm_instance)
        X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
        
        explanation = explainer(X_sample, check_additivity=False)
        shap_values = explanation.values

        df_imp = pd.DataFrame(
            {"feature": X_test.columns, "shap_importance": np.abs(shap_values).mean(axis=0)}
        ).sort_values("shap_importance", ascending=False)
        
        df_imp["is_dead_feature"] = (df_imp["shap_importance"] < 0.005).map({True: "Sim", False: "Não"})
        df_imp.to_csv("shap_importance.csv", index=False)
        mlflow.log_artifact("shap_importance.csv", "explainability")

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("Impacto Global das Features na Previsão Eólica (SHAP)", fontsize=14, pad=20)
        fig.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        # Validar se wind_temp_ratio existe no dataset (pode gerar erro dependendo da amostra)
        if "wind_temp_ratio" in X_sample.columns:
            shap.dependence_plot("wind_temp_ratio", shap_values, X_sample, ax=ax, show=False)
            plt.title("Curva de Resposta: Wind-Temp Ratio vs Impacto no FC", fontsize=14, pad=20)
            fig.savefig("shap_dependence.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact("shap_dependence.png", "explainability")
        
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation[0], show=False)
        plt.title("Explicabilidade Individual de uma Hora Específica", fontsize=14, pad=20)
        fig.savefig("shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact("shap_summary.png", "explainability")
        mlflow.log_artifact("shap_waterfall.png", "explainability")

        # ==============================================================================
        # 🛡️ REGISTRO DO MODELO COM TRUSTED TYPES
        # ==============================================================================
        trusted_types = [
            "collections.OrderedDict",
            "lightgbm.basic.Booster",
            "lightgbm.sklearn.LGBMRegressor",
            "sklearn.tree._tree.Tree",
            "sklearn.utils._bunch.Bunch",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBRegressor",
            "sklearn.linear_model._base.LinearRegression",
            "sklearn.model_selection._split.TimeSeriesSplit",
            (
                "energy_mlops.models.temporal_stacking."
                "TemporalStackingRegressor"
            ),
        ]

        signature = infer_signature(X_test, test_preds_fc)


        mlflow.sklearn.log_model(
            sk_model=ensemble,
            name="model",
            registered_model_name=model_name,
            skops_trusted_types=trusted_types,
            signature=signature,
            input_example=X_test.head(1),
        )
        
        return run_id, float(test_mae_mw)