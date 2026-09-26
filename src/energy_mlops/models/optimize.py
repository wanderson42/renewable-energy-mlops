import argparse
import os

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from energy_mlops.config import settings


def load_optimization_data(train_file: str):
    train_path = f"s3://{settings.RUSTFS_BUCKET}/gold/{train_file}"
    print(f"📖 A ler dados de Treino para Otimização: {train_path}")
    
    df_train = pd.read_parquet(train_path, storage_options=settings.storage_options)
    
    if "date" in df_train.columns:
        df_train["date"] = pd.to_datetime(df_train["date"])
        df_train.sort_values("date", inplace=True)
        df_train.reset_index(drop=True, inplace=True)

    drop_cols = ["wind_generation_mw", "target_fc", "date", "capacidade_mw"]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    return df_train[feature_cols], df_train["target_fc"]


def objective(trial, X, y, n_splits):
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

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_maes = []
    oof_r2s = []

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

        # Nota: Usamos média simples como proxy rápido (heuristic approach) para o StackingRegressor oficial.
        pred_ensemble = (pred_lgb + pred_xgb + pred_rf) / 3.0
        
        fold_mae = mean_absolute_error(y_val, pred_ensemble)
        fold_r2 = r2_score(y_val, pred_ensemble)
        
        oof_maes.append(fold_mae)
        oof_r2s.append(fold_r2)

    # Calculando estatísticas extras para rastreamento
    mean_mae = float(np.mean(oof_maes))
    std_mae = float(np.std(oof_maes))
    mean_r2 = float(np.mean(oof_r2s))
    
    # Registrando atributos extras no trial atual para resgatar no final
    trial.set_user_attr("cv_mae_std", std_mae)
    trial.set_user_attr("cv_r2_mean", mean_r2)

    return mean_mae


def run_optimization(train_file: str, n_trials: int = 20, n_splits: int = 3):
    X_train, y_train = load_optimization_data(train_file)
    
    print(f"🚀 Iniciando otimização do Ensemble via Optuna com {n_trials} trials e {n_splits} CV splits...")
    study = optuna.create_study(direction="minimize", study_name="Ensemble_Optimization")
    study.optimize(lambda trial: objective(trial, X_train, y_train, n_splits), n_trials=n_trials)

    best_trial = study.best_trial
    
    print("\n✅ Otimização concluída!")
    print(f"🏆 Melhor MAE (Target FC) validado no tempo: {best_trial.value:.5f}")
    
    best_params_raw = best_trial.params
    
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
        # Loga os parâmetros achatados na raiz da UI do MLflow para habilitar filtros/gráficos comparativos
        mlflow.log_params(best_params_raw)
        mlflow.log_param("cv_n_splits", n_splits)
        
        # Log das métricas globais e de variância
        mlflow.log_metric("best_cv_mae_mean", best_trial.value)
        mlflow.log_metric("best_cv_mae_std", best_trial.user_attrs.get("cv_mae_std", 0.0))
        mlflow.log_metric("best_cv_r2_mean", best_trial.user_attrs.get("cv_r2_mean", 0.0))
        
        # Salva o dicionário como um arquivo JSON diretamente no RustFS (Mantido para o train_ensemble.py)
        mlflow.log_dict(final_params, "best_ensemble_params.json")
        
    print("\n📊 Resumo dos Hiperparâmetros Otimizados:")
    print(f"   🟢 LightGBM: {final_params['lgb_params']}")
    print(f"   🔵 XGBoost: {final_params['xgb_params']}")
    print(f"   🟣 RandomForest: {final_params['rf_params']}")
    print("💾 Hiperparâmetros rastreados e salvos com sucesso no MLflow/RustFS!")


if __name__ == "__main__":
    os.environ["AWS_ACCESS_KEY_ID"] = settings.RUSTFS_ROOT_USER
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.RUSTFS_ROOT_PASSWORD
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.RUSTFS_ENDPOINT
    os.environ["AWS_ENDPOINT_URL"] = settings.RUSTFS_ENDPOINT

    parser = argparse.ArgumentParser(description="Otimização Optuna para LightGBM, XGBoost e RandomForest.")
    parser.add_argument("--train-file", type=str, default="dataset_renewable_energy_2024_1_2025_12.parquet")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=3, help="Número de quebras na validação cruzada temporal (TimeSeriesSplit)")
    args = parser.parse_args()

    run_optimization(args.train_file, args.trials, args.n_splits)

'''
# ==================================================================
# EXEMPLOS DE EXECUÇÃO VIA TERMINAL BASH
# ==================================================================

poetry run python src/energy_mlops/models/optimize.py \
    --train-file dataset_renewable_energy_2024_01_2025_12.parquet \
    --trials 30 \
    --n-splits 3
'''