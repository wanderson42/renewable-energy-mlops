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
    best_params: dict,
    optimization_run_id: str,
) -> tuple[str, float]:
    """
    Treina o modelo, registra todos os metadados (SHAP, trusted_types, métricas ricas) 
    e retorna o ID da Run e o MAE para o orquestrador (Prefect).
    """
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

    with mlflow.start_run(run_name="Stacking_LGB_XGB_RF_Bahia") as run:

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

        # 1. Recebe os parâmetros diretamente da execução de otimização
        lgb_params = best_params.get("lgb_params", {})
        xgb_params = best_params.get("xgb_params", {})
        rf_params = best_params.get("rf_params", {})


        required_param_groups = {
            "lgb_params": lgb_params,
            "xgb_params": xgb_params,
            "rf_params": rf_params,
        }

        missing_groups = [
            name
            for name, params in required_param_groups.items()
            if not params
        ]

        if missing_groups:
            raise ValueError(
                f"Parâmetros de otimização ausentes: {missing_groups}. "
                f"Optimization Run: {optimization_run_id}"
            )

        mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items()})
        mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})
        mlflow.log_params({f"rf_{k}": v for k, v in rf_params.items()})

        lgb_model = lgb.LGBMRegressor(**lgb_params)
        xgb_model = XGBRegressor(**xgb_params)
        rf_model = RandomForestRegressor(**rf_params)

        meta_learner = LinearRegression(
            positive=True,
            fit_intercept=False,
        )

        cv_strategy = TimeSeriesSplit(
            n_splits=5,
        )

        ensemble = TemporalStackingRegressor(
            estimators=[
                ("lgbm", lgb_model),
                ("xgboost", xgb_model),
                ("rf", rf_model),
            ],
            final_estimator=meta_learner,
            cv=cv_strategy,
        )

        print(f"🚀 A treinar Stacking Ensemble ({X_train.shape[0]} amostras, {num_features_challenger} features)...")
        ensemble.fit(X_train, y_train_fc)

        meta_coefs = ensemble.final_estimator_.coef_
        print(f"🧠 Pesos OOF aprendidos pelo Meta-Learner (LGB, XGB, RF): {meta_coefs}")

        mlflow.log_params(
            {f"meta_coef_{name}": float(coef) for name, coef in zip(["lgbm", "xgboost", "rf"], meta_coefs)}
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

        meta_weights = {}
        if hasattr(ensemble.final_estimator_, "coef_"):
            for (name, _), coef in zip(ensemble.named_estimators_.items(), ensemble.final_estimator_.coef_):
                meta_weights[name] = round(float(coef), 4)

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
        
        arch_str = f"{type(ensemble).__name__} ({', '.join(meta_weights.keys())}) -> {type(ensemble.final_estimator_).__name__}"
        mlflow.set_tag("architecture_str", arch_str)
        mlflow.set_tag("meta_weights_json", json.dumps(meta_weights))

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
        model_name = "ensemble_lgb_xgb_rf_bahia"

        mlflow.sklearn.log_model(
            sk_model=ensemble,
            name="model",
            registered_model_name=model_name,
            skops_trusted_types=trusted_types,
            signature=signature,
            input_example=X_test.head(1),
        )
        
        return run_id, float(test_mae_mw)