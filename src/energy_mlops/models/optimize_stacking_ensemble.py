import lightgbm as lgb
import mlflow
import mlflow.data
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from energy_mlops.config import settings


def prepare_optimization_data(df_train: pd.DataFrame):
    """Separa as features e o target a partir do DataFrame fornecido pelo orquestrador."""
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

        pred_ensemble = (pred_lgb + pred_xgb + pred_rf) / 3.0
        
        fold_mae = mean_absolute_error(y_val, pred_ensemble)
        fold_r2 = r2_score(y_val, pred_ensemble)
        
        oof_maes.append(fold_mae)
        oof_r2s.append(fold_r2)

    mean_mae = float(np.mean(oof_maes))
    std_mae = float(np.std(oof_maes))
    mean_r2 = float(np.mean(oof_r2s))
    
    trial.set_user_attr("cv_mae_std", std_mae)
    trial.set_user_attr("cv_r2_mean", mean_r2)

    return mean_mae

# Assinatura atualizada para receber o train_file
def run_optimization(df_train: pd.DataFrame, train_file: str, n_trials: int = 20, n_splits: int = 3) -> dict:
    """
    Contrato MLOps: Recebe o df_train e seu nome original do orquestrador, 
    roda Optuna e salva o melhor JSON e a linhagem dos dados no MLflow.
    """
    X_train, y_train = prepare_optimization_data(df_train)
    
    print(f"🚀 Iniciando otimização do Ensemble via Optuna com {n_trials} trials e {n_splits} CV splits...")
    study = optuna.create_study(direction="minimize", study_name="Ensemble_Optimization")
    study.optimize(lambda trial: objective(trial, X_train, y_train, n_splits), n_trials=n_trials)

    best_trial = study.best_trial
    best_params_raw = best_trial.params
    
    final_params = {
        "lgb_params": {k.replace("lgb_", ""): v for k, v in best_params_raw.items() if k.startswith("lgb_")},
        "xgb_params": {k.replace("xgb_", ""): v for k, v in best_params_raw.items() if k.startswith("xgb_")},
        "rf_params": {k.replace("rf_", ""): v for k, v in best_params_raw.items() if k.startswith("rf_")}
    }

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    experiment_name = "wind_power_optimization_bahia"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="Optuna_Best_Trial"):
        # 🟢 RASTREABILIDADE INSERIDA:
        mlflow.log_input(mlflow.data.from_pandas(df=X_train, source=f"s3://{settings.RUSTFS_BUCKET}/gold/{train_file}", name=train_file.replace(".parquet", "")), context="training")
        
        mlflow.log_params(best_params_raw)
        mlflow.log_param("cv_n_splits", n_splits)
        mlflow.log_metric("best_cv_mae_mean", best_trial.value)
        mlflow.log_metric("best_cv_mae_std", best_trial.user_attrs.get("cv_mae_std", 0.0))
        mlflow.log_dict(final_params, "best_ensemble_params.json")
        
    print("💾 Hiperparâmetros otimizados salvos com sucesso no MLflow!")
    return final_params