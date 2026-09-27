# src/energy_mlops/models/train_examples.py
"""
Exemplos de implementação que respeitam o contrato ModelTrainer.
Estes modelos demonstram como adicionar novos algoritmos ao pipeline agnóstico.
"""

import time

import mlflow
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


# ==============================================================================
# EXEMPLO 1: MOCK / DUMMY TRAINER (Para testes rápidos em CI/CD)
# ==============================================================================
def train_mock_dummy(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    train_file: str, 
    test_file: str
) -> tuple[str, float]:
    """
    Simula um treinamento instantâneo para testar a orquestração do Prefect 
    e o Quality Gate sem gastar recursos de CPU/GPU.
    """
    raise NotImplementedError("⚠️ ALERTA: Esta função é apenas um exemplo de contrato e não deve ser executada!")
    
    print("🧪 [MOCK] Executando treinamento fictício de teste...")
    time.sleep(0.5)
    
    mlflow.set_experiment("wind_power_forecasting_bahia")
    with mlflow.start_run(run_name="Mock_Dummy_Run") as run:
        mlflow.log_param("model_type", "DummyMock")
        mlflow.log_param("num_features", 5)
        mlflow.log_metric("oot_mae_mw", 12.50)
        mlflow.log_metric("oot_nmae_pct_2026", 2.10)
        
        run_id = run.info.run_id
        
    return run_id, 12.50


# ==============================================================================
# EXEMPLO 2: REGRESSÃO LINEAR / RIDGE (Modelo Simples e Rápido)
# ==============================================================================
def train_ridge_regression(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    train_file: str, 
    test_file: str
) -> tuple[str, float]:
    """
    Treina uma Regressão Ridge simples como Baseline.
    Demonstra a extração de features e a API clássica do scikit-learn.
    """
    raise NotImplementedError("⚠️ ALERTA: Esta função é apenas um exemplo de contrato e não deve ser executada!")
    
    drop_cols = ["wind_generation_mw", "target_fc", "date", "capacidade_mw"]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]
    
    X_train, y_train = df_train[feature_cols].fillna(0), df_train["target_fc"]
    X_test, y_test = df_test[feature_cols].fillna(0), df_test["target_fc"]
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    
    mlflow.set_experiment("wind_power_forecasting_bahia")
    with mlflow.start_run(run_name="Ridge_Baseline") as run:
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_metric("oot_mae_mw", mae)
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        run_id = run.info.run_id
        
    return run_id, mae


# ==============================================================================
# EXEMPLO 3: REDE NEURAL PROFUNDA (PyTorch / Deep Learning)
# ==============================================================================
def train_pytorch_nn(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    train_file: str, 
    test_file: str
) -> tuple[str, float]:
    """
    Estrutura para treinamento de Redes Neurais.
    Demonstra a criação de Tensores, DataLoaders e log com mlflow.pytorch.
    """
    raise NotImplementedError("⚠️ ALERTA: Esta função é apenas um exemplo de contrato e não deve ser executada!")
    
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    drop_cols = ["wind_generation_mw", "target_fc", "date", "capacidade_mw"]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]
    
    # 1. Preparação de Tensores
    X_tr = torch.tensor(df_train[feature_cols].fillna(0).values, dtype=torch.float32)
    y_tr = torch.tensor(df_train["target_fc"].values, dtype=torch.float32).unsqueeze(1)
    X_te = torch.tensor(df_test[feature_cols].fillna(0).values, dtype=torch.float32)
    y_te = torch.tensor(df_test["target_fc"].values, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. Definição da Arquitetura
    class WindMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x)

    model = WindMLP(input_dim=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss() # MAE Loss
    
    # 3. Loop de Treino
    model.train()
    for epoch in range(5):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
    # 4. Avaliação
    model.eval()
    with torch.no_grad():
        preds = model(X_te)
        mae = float(criterion(preds, y_te).item())
        
    # 5. MLflow Log
    mlflow.set_experiment("wind_power_forecasting_bahia")
    with mlflow.start_run(run_name="PyTorch_MLP_Wind") as run:
        mlflow.log_param("model_type", "PyTorch_MLP")
        mlflow.log_param("epochs", 5)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_metric("oot_mae_mw", mae)
        mlflow.pytorch.log_model(model, artifact_path="model")
        
        run_id = run.info.run_id

    return run_id, mae