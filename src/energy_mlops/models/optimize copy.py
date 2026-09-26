import argparse
import os

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from energy_mlops.config import settings


def load_optimization_data(train_file: str):
    train_path = f"s3://{settings.MINIO_BUCKET}/gold/{train_file}"
    print(f"📖 A ler dados de Treino para Otimização: {train_path}")
    
    df_train = pd.read_parquet(train_path, storage_options=settings.storage_options)
    
    if "date" in df_train.columns:
        df_train["date"] = pd.to_datetime(df_train["date"])
        df_train.sort_values("date", inplace=True)
        df_train.reset_index(drop=True, inplace=True)

    drop_cols = ["wind_generation_mw", "target_fc", "date", "capacidade_mw"]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    return df_train[feature_cols], df_train["target_fc"]


def objective(trial, X, y):
    lgb_params = {
        "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("lgb_learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("lgb_num_leaves", 20, 100),
        "max_depth": trial.suggest_int("lgb_max_depth", 3, 10),
        "objective": "regression",
        "metric": "mae",
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1
    }

    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
        "objective": "reg:absoluteerror",
        "random_state": 42,
        "n_jobs": -1
    }

    rf_params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 300),
        "max_depth": trial.suggest_int("rf_max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
        "random_state": 42,
        "n_jobs": -1
    }

    tscv = TimeSeriesSplit(n_splits=3)
    oof_maes = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(X_tr, y_tr)
        pred_lgb = model_lgb.predict(X_val)

        model_xgb = XGBRegressor(**xgb_params)
        model_xgb.fit(X_tr, y_tr)
        pred_xgb = model_xgb.predict(X_val)

        model_rf = RandomForestRegressor(**rf_params)
        model_rf.fit(X_tr, y_tr)
        pred_rf = model_rf.predict(X_val)

        pred_ensemble = (pred_lgb + pred_xgb + pred_rf) / 3.0
        fold_mae = mean_absolute_error(y_val, pred_ensemble)
        oof_maes.append(fold_mae)

    return sum(oof_maes) / len(oof_maes)


def run_optimization(train_file: str, n_trials: int = 20):
    X_train, y_train = load_optimization_data(train_file)
    
    print(f"🚀 Iniciando otimização do Ensemble via Optuna com {n_trials} trials...")
    study = optuna.create_study(direction="minimize", study_name="Ensemble_Optimization")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    print("\n✅ Otimização concluída!")
    print(f"🏆 Melhor MAE (Target FC) validado no tempo: {study.best_value:.5f}")
    
    best_params_raw = study.best_params
    
    final_params = {
        "lgb_params": {k.replace("lgb_", ""): v for k, v in best_params_raw.items() if k.startswith("lgb_")},
        "xgb_params": {k.replace("xgb_", ""): v for k, v in best_params_raw.items() if k.startswith("xgb_")},
        "rf_params": {k.replace("rf_", ""): v for k, v in best_params_raw.items() if k.startswith("rf_")}
    }

    # ============Rastreamento pelo MLflow======================
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    experiment_name = "wind_power_optimization_bahia"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="Optuna_Best_Trial"):
        mlflow.log_metric("best_cv_mae", study.best_value)
        # Salva o dicionário como um arquivo JSON diretamente no MinIO!
        mlflow.log_dict(final_params, "best_ensemble_params.json")
        
    print("\n📊 Resumo dos Hiperparâmetros Otimizados:")
    print(f"   🟢 LightGBM: {final_params['lgb_params']}")
    print(f"   🔵 XGBoost: {final_params['xgb_params']}")
    print(f"   🟣 RandomForest: {final_params['rf_params']}")
    print("💾 Hiperparâmetros rastreados e salvos com sucesso no MLflow/MinIO!")


if __name__ == "__main__":
    os.environ["AWS_ACCESS_KEY_ID"] = settings.MINIO_ROOT_USER
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.MINIO_ROOT_PASSWORD
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MINIO_ENDPOINT
    os.environ["AWS_ENDPOINT_URL"] = settings.MINIO_ENDPOINT

    parser = argparse.ArgumentParser(description="Otimização Optuna para LightGBM, XGBoost e RandomForest.")
    parser.add_argument("--train-file", type=str, default="dataset_renewable_energy_2024_1_2025_12.parquet")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    run_optimization(args.train_file, args.trials)

'''
# ==================================================================
# EXEMPLOS DE EXECUÇÃO VIA TERMINAL BASH
# ==================================================================

poetry run python src/energy_mlops/models/optimize.py \
    --train-file dataset_renewable_energy_2024_1_2025_12.parquet \
    --trials 30
'''     