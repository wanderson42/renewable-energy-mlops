import argparse
import os

import lightgbm as lgb
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from energy_mlops.config import settings


def load_data(train_file: str, test_file: str) -> tuple:
    train_path = f"s3://{settings.MINIO_BUCKET}/gold/{train_file}"
    test_path = f"s3://{settings.MINIO_BUCKET}/gold/{test_file}"
    
    print(f"📖 A ler Treino de: {train_path}")
    df_train = pd.read_parquet(train_path, storage_options=settings.storage_options)
    print(f"📖 A ler Teste (OOT) de: {test_path}")
    df_test = pd.read_parquet(test_path, storage_options=settings.storage_options)
    
    # Ordenação temporal defensiva
    for df in [df_train, df_test]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)

    drop_cols = ["wind_generation_mw", "target_fc", "date", "capacidade_mw"]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    return (
        df_train[feature_cols], df_test[feature_cols], 
        df_train["target_fc"], df_test["target_fc"],
        df_train["wind_generation_mw"], df_test["wind_generation_mw"],
        df_train["capacidade_mw"], df_test["capacidade_mw"]
    )

def train_baseline(train_file: str, test_file: str):
    X_train, X_test, y_train_fc, y_test_fc, y_train_mw, y_test_mw, cap_train, cap_test = load_data(train_file, test_file)

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    experiment_name = "wind_power_forecasting_bahia"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
        print(f"♻️ Experimento '{experiment_name}' restaurado do lixo.")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="LightGBM_Baseline_Bahia"):
        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "random_state": 42
        }
        mlflow.log_params(params)

        print(f"🚀 A treinar modelo Baseline LightGBM no Fator de Capacidade ({X_train.shape[0]} amostras, {X_train.shape} features)...")
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train_fc)

# --- Desnormalização e Avaliação no Treino ---
        train_preds_fc = model.predict(X_train)
        train_preds_mw = train_preds_fc * cap_train
        
        train_mae_mw = mean_absolute_error(y_train_mw, train_preds_mw)
        train_rmse_mw = root_mean_squared_error(y_train_mw, train_preds_mw)
        
        # Métrica Normalizada (nMAE em %)
        train_nmae_pct = (train_mae_mw / cap_train.max()) * 100
        train_mae_fc_pct = mean_absolute_error(y_train_fc, train_preds_fc) * 100

        # --- Desnormalização e Avaliação no Teste (OOT 2026) ---
        test_preds_fc = model.predict(X_test)
        test_preds_mw = test_preds_fc * cap_test
        
        test_mae_mw = mean_absolute_error(y_test_mw, test_preds_mw)
        test_rmse_mw = root_mean_squared_error(y_test_mw, test_preds_mw)
        
        # Métrica Normalizada (nMAE em %)
        test_nmae_pct = (test_mae_mw / cap_test.max()) * 100
        test_mae_fc_pct = mean_absolute_error(y_test_fc, test_preds_fc) * 100

        # --- Logging no MLflow ---
        mlflow.log_metric("train_mae_mw", train_mae_mw)
        mlflow.log_metric("train_rmse_mw", train_rmse_mw)
        mlflow.log_metric("train_nmae_pct", train_nmae_pct)
        mlflow.log_metric("train_mae_fc_pct", train_mae_fc_pct)

        mlflow.log_metric("oot_mae_mw_2026", test_mae_mw)
        mlflow.log_metric("oot_rmse_mw_2026", test_rmse_mw)
        mlflow.log_metric("oot_nmae_pct_2026", test_nmae_pct)
        mlflow.log_metric("oot_mae_fc_pct_2026", test_mae_fc_pct)

        trusted_types = ["collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMRegressor",
                          "sklearn.tree._tree.Tree", "sklearn.utils._bunch.Bunch", "xgboost.core.Booster', 'xgboost.sklearn.XGBRegressor"]

        mlflow.lightgbm.log_model(
            lgb_model=model,
            name="model",
            skops_trusted_types=trusted_types,
            registered_model_name="baseline_lgbm_model_bahia"
        )

        print("✅ Treino concluído!")
        print(f"📊 Desempenho TREINO: MAE: {train_mae_mw:.2f} MW | nMAE: {train_nmae_pct:.2f}% | RMSE: {train_rmse_mw:.2f} MW")
        print(f"🎯 Desempenho OOT:    MAE: {test_mae_mw:.2f} MW | nMAE: {test_nmae_pct:.2f}% | RMSE: {test_rmse_mw:.2f} MW")

if __name__ == "__main__":
    os.environ["AWS_ACCESS_KEY_ID"] = settings.MINIO_ROOT_USER
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.MINIO_ROOT_PASSWORD
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MINIO_ENDPOINT
    os.environ["AWS_ENDPOINT_URL"] = settings.MINIO_ENDPOINT

    parser = argparse.ArgumentParser(description="Treino do modelo Baseline LightGBM.")
    parser.add_argument("--train-file", type=str, default="dataset_renewable_energy_2025_1_2025_12.parquet")
    parser.add_argument("--test-file", type=str, default="dataset_renewable_energy_2026_1_2026_8.parquet")
    args = parser.parse_args()

    train_baseline(args.train_file, args.test_file)