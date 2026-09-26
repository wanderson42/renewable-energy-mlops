import argparse
import json
import os
import tempfile

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from energy_mlops.config import settings


def load_best_ensemble_params_from_mlflow() -> tuple[dict, dict, dict]:
    """Busca o JSON com os melhores parâmetros diretamente do artefato do MLflow."""
    default_lgb = {
        "objective": "regression", "metric": "mae", "boosting_type": "gbdt",
        "n_estimators": 400, "learning_rate": 0.03, "num_leaves": 45, "max_depth": 5,
        "random_state": 42, "verbosity": -1, "n_jobs": -1,
    }
    default_xgb = {
        "objective": "reg:absoluteerror", "n_estimators": 400, "learning_rate": 0.03,
        "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1,
    }
    default_rf = {
        "n_estimators": 200, "max_depth": 12, "min_samples_split": 10, "random_state": 42, "n_jobs": -1,
    }

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    opt_exp_name = "wind_power_optimization_bahia"
    
    experiment = client.get_experiment_by_name(opt_exp_name)
    if experiment:
        # Busca a última run executada com sucesso na Otimização
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        if runs:
            run_id = runs[0].info.run_id
            print(f"☁️ A baixar parâmetros otimizados do MLflow (Run ID: {run_id})...")
            try:
                # Usa diretório temporário para baixar o artefato (não suja o disco)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_path = client.download_artifacts(run_id, "best_ensemble_params.json", tmp_dir)
                    with open(local_path, "r") as f:
                        loaded = json.load(f)
                        
                    lgb_p = loaded.get("lgb_params", {})
                    xgb_p = loaded.get("xgb_params", {})
                    rf_p = loaded.get("rf_params", {})

                    lgb_p.update({"objective": "regression", "metric": "mae", "random_state": 42, "verbosity": -1, "n_jobs": -1})
                    xgb_p.update({"objective": "reg:absoluteerror", "random_state": 42, "n_jobs": -1})
                    rf_p.update({"random_state": 42, "n_jobs": -1})

                    return lgb_p, xgb_p, rf_p
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ Erro ao baixar artefato do MLflow: {e}. A utilizar parâmetros padrão.")
        else:
            print("⚠️ Nenhuma Run de otimização encontrada. A utilizar parâmetros padrão.")
    else:
        print(f"⚠️ Experimento '{opt_exp_name}' não encontrado. A utilizar parâmetros padrão.")
        
    return default_lgb, default_xgb, default_rf


def load_data(train_file: str, test_file: str) -> tuple:
    train_path = f"s3://{settings.RUSTFS_BUCKET}/gold/{train_file}"
    test_path = f"s3://{settings.RUSTFS_BUCKET}/gold/{test_file}"

    print(f"📖 A ler Treino de: {train_path}")
    df_train = pd.read_parquet(train_path, storage_options=settings.storage_options)
    print(f"📖 A ler Teste (OOT Jan-Ago 2026) de: {test_path}")
    df_test = pd.read_parquet(test_path, storage_options=settings.storage_options)

    for df in [df_train, df_test]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)

    drop_cols = ["wind_generation_mw", "target_fc", "date", "capacidade_mw"]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    return (
        df_train[feature_cols], df_test[feature_cols], df_train["target_fc"], df_test["target_fc"],
        df_train["wind_generation_mw"], df_test["wind_generation_mw"], df_train["capacidade_mw"], df_test["capacidade_mw"],
    )


def train_ensemble(train_file: str, test_file: str):
    X_train, X_test, y_train_fc, y_test_fc, y_train_mw, y_test_mw, cap_train, cap_test = load_data(train_file, test_file)

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    experiment_name = "wind_power_forecasting_bahia"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Stacking_LGB_XGB_RF_Bahia"):
        mlflow.log_input(mlflow.data.from_pandas(df=X_train, source=f"s3://{settings.RUSTFS_BUCKET}/gold/{train_file}", name=train_file.replace(".parquet", "")), context="training")
        mlflow.log_input(mlflow.data.from_pandas(df=X_test, source=f"s3://{settings.RUSTFS_BUCKET}/gold/{test_file}", name=test_file.replace(".parquet", "")), context="testing")

        num_features_challenger = len(X_train.columns)
        mlflow.log_param("num_features", num_features_challenger)

        # 1. Carrega parâmetros consultando o MLflow
        lgb_params, xgb_params, rf_params = load_best_ensemble_params_from_mlflow()

        mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items()})
        mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})
        mlflow.log_params({f"rf_{k}": v for k, v in rf_params.items()})

        lgb_model = lgb.LGBMRegressor(**lgb_params)
        xgb_model = XGBRegressor(**xgb_params)
        rf_model = RandomForestRegressor(**rf_params)

        meta_learner = LinearRegression(positive=True, fit_intercept=False)
        cv_strategy = KFold(n_splits=5, shuffle=False)

        ensemble = StackingRegressor(
            estimators=[("lgbm", lgb_model), ("xgboost", xgb_model), ("rf", rf_model)],
            final_estimator=meta_learner,
            cv=cv_strategy,
            n_jobs=-1,
        )

        print(f"🚀 A treinar Stacking Ensemble ({X_train.shape[0]} amostras, {num_features_challenger} features)...")
        ensemble.fit(X_train, y_train_fc)

        meta_coefs = ensemble.final_estimator_.coef_
        print(f"🧠 Pesos OOF aprendidos pelo Meta-Learner (LGB, XGB, RF): {meta_coefs}")
        mlflow.log_params({f"meta_coef_{name}": float(coef) for name, coef in zip(["lgbm", "xgboost", "rf"], meta_coefs)})

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
        mlflow.log_metric("oot_mae_mw_2026", test_mae_mw)
        mlflow.log_metric("oot_nmae_pct_2026", test_nmae_pct)
        mlflow.log_metric("oot_mae_fc_pct_2026", test_mae_fc_pct)
        mlflow.log_metric("oot_r2_score_2026", test_r2)

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

        # Salva o JSON na raiz dos artefatos
        mlflow.log_dict(model_summary, "model_summary.json")
        
        # Tags de rastreabilidade na UI do MLflow
        arch_str = f"{type(ensemble).__name__} ({', '.join(meta_weights.keys())}) -> {type(ensemble.final_estimator_).__name__}"
        mlflow.set_tag("architecture_str", arch_str)
        mlflow.set_tag("meta_weights_json", json.dumps(meta_weights))

        # Explicabilidade SHAP
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
        shap.dependence_plot("wind_temp_ratio", shap_values, X_sample, ax=ax, show=False)
        plt.title("Curva de Resposta: Wind-Temp Ratio vs Impacto no FC", fontsize=14, pad=20)
        fig.savefig("shap_dependence.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation[0], show=False)
        plt.title("Explicabilidade Individual de uma Hora Específica", fontsize=14, pad=20)
        fig.savefig("shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        for img in ["shap_summary.png", "shap_dependence.png", "shap_waterfall.png"]:
            mlflow.log_artifact(img, "explainability")

        # Registro do Modelo
        trusted_types = [
            "collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMRegressor",
            "sklearn.tree._tree.Tree", "sklearn.utils._bunch.Bunch", "xgboost.core.Booster",
            "xgboost.sklearn.XGBRegressor", "sklearn.linear_model._base.LinearRegression",
            "sklearn.model_selection._split.KFold"
        ]

        signature = infer_signature(X_test, test_preds_fc)
        model_name = "ensemble_lgb_xgb_rf_bahia"

        model_info = mlflow.sklearn.log_model(
            sk_model=ensemble,
            name="model",
            registered_model_name=model_name,
            skops_trusted_types=trusted_types,
            signature=signature,
        )
        challenger_version = model_info.registered_model_version

        # Quality Gate
        print("\n🛡️ INICIANDO QUALITY GATE (Champion vs Challenger)...")
        promote_to_champion = False
        try:
            champion_info = client.get_model_version_by_alias(model_name, "champion")
            champ_run = client.get_run(champion_info.run_id)
            champ_mae = champ_run.data.metrics.get("oot_mae_mw_2026")
            if test_mae_mw < champ_mae:
                print("🎉 APROVADO! Challenger superou o Champion.")
                promote_to_champion = True
            else:
                print("❌ REPROVADO! Modelo degradou.")
        except MlflowException:
            print("ℹ️ Nenhum '@champion' encontrado. Promovendo primeiro Campeão!")
            promote_to_champion = True

        if promote_to_champion:
            client.set_registered_model_alias(model_name, "champion", challenger_version)

if __name__ == "__main__":
    os.environ["AWS_ACCESS_KEY_ID"] = settings.RUSTFS_ROOT_USER
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.RUSTFS_ROOT_PASSWORD
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.RUSTFS_ENDPOINT
    os.environ["AWS_ENDPOINT_URL"] = settings.RUSTFS_ENDPOINT

    parser = argparse.ArgumentParser(description="Treino do Stacking Ensemble (LGB+XGB+RF).")
    parser.add_argument("--train-file", type=str, default="dataset_renewable_energy_2024_1_2025_12.parquet")
    parser.add_argument("--test-file", type=str, default="dataset_renewable_energy_2026_1_2026_8.parquet")
    args = parser.parse_args()

    train_ensemble(args.train_file, args.test_file)

'''
# ==================================================================
# EXEMPLOS DE EXECUÇÃO VIA TERMINAL BASH
# ==================================================================

poetry run python src/energy_mlops/models/train_ensemble.py \
    --train-file dataset_renewable_energy_2024_01_2025_12.parquet \
    --test-file dataset_renewable_energy_2026_01_2026_08.parquet
'''     