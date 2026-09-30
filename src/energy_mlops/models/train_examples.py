# src/energy_mlops/models/train_examples.py
"""
Exemplos de ModelTrainer compatíveis com a infraestrutura MLOps.

Este módulo demonstra como implementar novos algoritmos de treinamento
sem acoplá-los ao Prefect.

Contrato esperado pelo orquestrador:

    trainer(
        df_train,
        df_test,
        train_file,
        test_file,
        best_params=None,
        optimization_run_id=None,
    ) -> tuple[str, float]

Requisitos para integração com o lifecycle atual:

1. usar o contrato central de features;
2. registrar ``num_features`` no MLflow;
3. registrar métricas com nomes canônicos;
4. registrar o modelo no Model Registry;
5. retornar ``(run_id, oot_mae_mw)``;
6. aceitar execução com ou sem etapa de otimização.

Os exemplos NÃO são registrados por padrão no TRAINER_REGISTRY.

                        ModelTrainer
                             │
              ┌──────────────┴──────────────┐
              │                             │
            Ridge                        PyTorch
              │                             │
      optimizer opcional             optimizer opcional
              │                             │
              └───────────┬─────────────────┘
                          │
                select_model_features()
                          │
                    predict FC
                          │
                  FC x capacidade
                          │
                      MW forecast
                          │
               métricas canônicas
                          │
                       MLflow
                          │
                  Model Registry
                          │
               (run_id, oot_mae_mw)
"""


from typing import Any

import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from energy_mlops.config import settings
from energy_mlops.data.feature_utils import select_model_features

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

EXPERIMENT_NAME = "wind_power_forecasting_bahia"

# Compatibilidade com o Model Registry atual.
#
# O nome ainda reflete a arquitetura histórica do Champion.
# Futuramente pode ser migrado para um nome orientado ao produto,
# por exemplo "wind_power_forecasting_bahia".
REGISTERED_MODEL_NAME = "ensemble_lgb_xgb_rf_bahia"


# ==============================================================================
# HELPERS DO CONTRATO
# ==============================================================================


def _validate_optimization_context(
    best_params: dict | None,
    optimization_run_id: str | None,
) -> None:
    """
    Garante consistência entre parâmetros e linhagem da otimização.

    O trainer pode operar:
    - sem optimizer: ambos ``None``;
    - com optimizer: ambos preenchidos.

    Estados parciais são considerados configuração inválida.
    """

    has_params = best_params is not None
    has_run_id = optimization_run_id is not None

    if has_params != has_run_id:
        raise ValueError(
            "Contexto de otimização inconsistente: "
            "'best_params' e 'optimization_run_id' "
            "devem ser fornecidos juntos ou ambos devem ser None."
        )


def _prepare_training_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Prepara features, targets e capacidade usando o contrato central.

    Nenhum trainer deve reconstruir sua própria lista de features.
    """

    X_train = select_model_features(
        df_train
    )

    X_test = select_model_features(
        df_test
    )

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


def _calculate_metrics(
    *,
    y_train_fc: pd.Series,
    y_test_fc: pd.Series,
    y_train_mw: pd.Series,
    y_test_mw: pd.Series,
    cap_train: pd.Series,
    cap_test: pd.Series,
    train_preds_fc: np.ndarray,
    test_preds_fc: np.ndarray,
) -> dict[str, float]:
    """
    Calcula as mesmas métricas usadas pelo trainer oficial.

    O modelo prevê fator de capacidade (FC). A previsão em MW é
    obtida multiplicando FC pela capacidade operacional correspondente.
    """

    if cap_train.max() <= 0:
        raise ValueError(
            "A capacidade máxima de treino deve ser maior que zero."
        )

    if cap_test.max() <= 0:
        raise ValueError(
            "A capacidade máxima OOT deve ser maior que zero."
        )

    train_preds_mw = (
        train_preds_fc
        * cap_train.to_numpy()
    )

    test_preds_mw = (
        test_preds_fc
        * cap_test.to_numpy()
    )

    train_mae_mw = float(
        mean_absolute_error(
            y_train_mw,
            train_preds_mw,
        )
    )

    train_nmae_pct = float(
        (
            train_mae_mw
            / cap_train.max()
        )
        * 100
    )

    train_mae_fc_pct = float(
        mean_absolute_error(
            y_train_fc,
            train_preds_fc,
        )
        * 100
    )

    oot_mae_mw = float(
        mean_absolute_error(
            y_test_mw,
            test_preds_mw,
        )
    )

    oot_nmae_pct = float(
        (
            oot_mae_mw
            / cap_test.max()
        )
        * 100
    )

    oot_mae_fc_pct = float(
        mean_absolute_error(
            y_test_fc,
            test_preds_fc,
        )
        * 100
    )

    oot_r2_score = float(
        r2_score(
            y_test_fc,
            test_preds_fc,
        )
    )

    return {
        "train_mae_mw": train_mae_mw,
        "train_nmae_pct": train_nmae_pct,
        "train_mae_fc_pct": train_mae_fc_pct,
        "oot_mae_mw": oot_mae_mw,
        "oot_nmae_pct": oot_nmae_pct,
        "oot_mae_fc_pct": oot_mae_fc_pct,
        "oot_r2_score": oot_r2_score,
    }


def _prepare_mlflow_experiment() -> None:
    """
    Configura o tracking e garante que o experimento esteja disponível.
    """

    mlflow.set_tracking_uri(
        settings.MLFLOW_TRACKING_URI
    )

    client = MlflowClient(
        tracking_uri=(
            settings.MLFLOW_TRACKING_URI
        )
    )

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if (
        experiment is not None
        and experiment.lifecycle_stage == "deleted"
    ):
        client.restore_experiment(
            experiment.experiment_id
        )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


def _log_dataset_lineage(
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_file: str,
    test_file: str,
) -> None:
    """
    Registra a linhagem dos datasets Gold utilizados pela Run.
    """

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

    mlflow.log_input(
        mlflow.data.from_pandas(
            df=X_test,
            source=(
                f"s3://{settings.RUSTFS_BUCKET}"
                f"/gold/{test_file}"
            ),
            name=test_file.replace(
                ".parquet",
                "",
            ),
        ),
        context="testing",
    )


def _log_common_metadata(
    *,
    model_type: str,
    num_features: int,
    metrics: dict[str, float],
    optimization_run_id: str | None,
) -> None:
    """
    Registra metadados obrigatórios para o lifecycle do modelo.
    """

    mlflow.log_param(
        "num_features",
        num_features,
    )

    mlflow.log_param(
        "model_type",
        model_type,
    )

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(
            metric_name,
            metric_value,
        )

    if optimization_run_id is not None:
        mlflow.set_tag(
            "optimization_run_id",
            optimization_run_id,
        )

        mlflow.set_tag(
            "optimization_used",
            "true",
        )

    else:
        mlflow.set_tag(
            "optimization_used",
            "false",
        )

    mlflow.set_tag(
        "trainer_example_source",
        "train_examples.py",
    )


# ==============================================================================
# EXEMPLO 1: RIDGE — TRAINER SEM OTIMIZAÇÃO OBRIGATÓRIA
# ==============================================================================


def train_ridge_regression(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    train_file: str,
    test_file: str,
    best_params: dict | None = None,
    optimization_run_id: str | None = None,
) -> tuple[str, float]:
    """
    Exemplo de ModelTrainer baseado em Ridge Regression.

    O trainer funciona normalmente sem optimizer:

        best_params=None
        optimization_run_id=None

    Caso uma estratégia de otimização específica para Ridge seja criada,
    ela pode produzir:

        {
            "ridge_params": {
                "alpha": ...
            }
        }

    O retorno segue o contrato exigido pelo orquestrador:

        (run_id, oot_mae_mw)
    """

    _validate_optimization_context(
        best_params,
        optimization_run_id,
    )

    (
        X_train,
        X_test,
        y_train_fc,
        y_test_fc,
        y_train_mw,
        y_test_mw,
        cap_train,
        cap_test,
    ) = _prepare_training_data(
        df_train,
        df_test,
    )

    ridge_params: dict[str, Any] = {
        "alpha": 1.0,
    }

    if best_params is not None:
        optimized_params = (
            best_params.get(
                "ridge_params",
                {},
            )
        )

        ridge_params.update(
            optimized_params
        )

    model = Ridge(
        **ridge_params
    )

    model.fit(
        X_train,
        y_train_fc,
    )

    train_preds_fc = model.predict(
        X_train
    )

    test_preds_fc = model.predict(
        X_test
    )

    metrics = _calculate_metrics(
        y_train_fc=y_train_fc,
        y_test_fc=y_test_fc,
        y_train_mw=y_train_mw,
        y_test_mw=y_test_mw,
        cap_train=cap_train,
        cap_test=cap_test,
        train_preds_fc=train_preds_fc,
        test_preds_fc=test_preds_fc,
    )

    _prepare_mlflow_experiment()

    with mlflow.start_run(
        run_name="Ridge_Baseline_Bahia"
    ) as run:
        run_id = run.info.run_id

        _log_dataset_lineage(
            X_train=X_train,
            X_test=X_test,
            train_file=train_file,
            test_file=test_file,
        )

        mlflow.log_params(
            {
                f"ridge_{name}": value
                for name, value
                in ridge_params.items()
            }
        )

        _log_common_metadata(
            model_type="Ridge",
            num_features=len(
                X_train.columns
            ),
            metrics=metrics,
            optimization_run_id=(
                optimization_run_id
            ),
        )

        mlflow.set_tag(
            "architecture_str",
            "Ridge",
        )

        signature = infer_signature(
            X_test,
            test_preds_fc,
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=(
                REGISTERED_MODEL_NAME
            ),
            signature=signature,
            input_example=X_test.head(1),
        )

    print(
        "✅ Treino Ridge concluído!"
    )

    print(
        "🔥 Desempenho OOT: "
        f"MAE={metrics['oot_mae_mw']:.2f} MW | "
        f"nMAE={metrics['oot_nmae_pct']:.2f}% | "
        f"MAE_FC={metrics['oot_mae_fc_pct']:.2f}% | "
        f"R²={metrics['oot_r2_score']:.4f}"
    )

    return (
        run_id,
        metrics["oot_mae_mw"],
    )


# ==============================================================================
# EXEMPLO 2: PYTORCH — TRAINER DE OUTRO ECOSSISTEMA
# ==============================================================================


def train_pytorch_nn(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    train_file: str,
    test_file: str,
    best_params: dict | None = None,
    optimization_run_id: str | None = None,
) -> tuple[str, float]:
    """
    Exemplo de ModelTrainer baseado em PyTorch.

    O objetivo deste exemplo é demonstrar que o contrato do orquestrador
    não depende da API scikit-learn.

    PyTorch é uma dependência opcional deste exemplo e não precisa fazer
    parte do ambiente principal enquanto este trainer não estiver ativo.

    Parâmetros opcionais esperados:

        {
            "nn_params": {
                "hidden_dim": 64,
                "epochs": 5,
                "batch_size": 64,
                "learning_rate": 0.001
            }
        }
    """

    try:
        import torch
        from torch import nn
        from torch.utils.data import (
            DataLoader,
            TensorDataset,
        )

    except ImportError as exc:
        raise RuntimeError(
            "O exemplo PyTorch requer a dependência opcional "
            "'torch'. Instale-a antes de executar este trainer."
        ) from exc

    _validate_optimization_context(
        best_params,
        optimization_run_id,
    )

    (
        X_train,
        X_test,
        y_train_fc,
        y_test_fc,
        y_train_mw,
        y_test_mw,
        cap_train,
        cap_test,
    ) = _prepare_training_data(
        df_train,
        df_test,
    )

    nn_params: dict[str, Any] = {
        "hidden_dim": 64,
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 0.001,
    }

    if best_params is not None:
        optimized_params = (
            best_params.get(
                "nn_params",
                {},
            )
        )

        nn_params.update(
            optimized_params
        )

    hidden_dim = int(
        nn_params["hidden_dim"]
    )

    epochs = int(
        nn_params["epochs"]
    )

    batch_size = int(
        nn_params["batch_size"]
    )

    learning_rate = float(
        nn_params["learning_rate"]
    )

    X_train_np = (
        X_train
        .astype("float32")
        .to_numpy()
    )

    X_test_np = (
        X_test
        .astype("float32")
        .to_numpy()
    )

    y_train_np = (
        y_train_fc
        .astype("float32")
        .to_numpy()
        .reshape(-1, 1)
    )

    X_train_tensor = torch.tensor(
        X_train_np,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train_np,
        dtype=torch.float32,
    )

    X_test_tensor = torch.tensor(
        X_test_np,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    class WindMLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
        ):
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(
                    input_dim,
                    hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    hidden_dim,
                    max(
                        hidden_dim // 2,
                        1,
                    ),
                ),
                nn.ReLU(),
                nn.Linear(
                    max(
                        hidden_dim // 2,
                        1,
                    ),
                    1,
                ),
            )

        def forward(self, x):
            return self.network(x)

    model = WindMLP(
        input_dim=len(
            X_train.columns
        ),
        hidden_dim=hidden_dim,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.L1Loss()

    model.train()

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            predictions = model(
                batch_x
            )

            loss = criterion(
                predictions,
                batch_y,
            )

            loss.backward()

            optimizer.step()

    model.eval()

    with torch.no_grad():
        train_preds_fc = (
            model(
                X_train_tensor
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )

        test_preds_fc = (
            model(
                X_test_tensor
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )

    metrics = _calculate_metrics(
        y_train_fc=y_train_fc,
        y_test_fc=y_test_fc,
        y_train_mw=y_train_mw,
        y_test_mw=y_test_mw,
        cap_train=cap_train,
        cap_test=cap_test,
        train_preds_fc=train_preds_fc,
        test_preds_fc=test_preds_fc,
    )

    _prepare_mlflow_experiment()

    with mlflow.start_run(
        run_name="PyTorch_MLP_Wind"
    ) as run:
        run_id = run.info.run_id

        _log_dataset_lineage(
            X_train=X_train,
            X_test=X_test,
            train_file=train_file,
            test_file=test_file,
        )

        mlflow.log_params(
            {
                f"nn_{name}": value
                for name, value
                in nn_params.items()
            }
        )

        _log_common_metadata(
            model_type="PyTorch_MLP",
            num_features=len(
                X_train.columns
            ),
            metrics=metrics,
            optimization_run_id=(
                optimization_run_id
            ),
        )

        mlflow.set_tag(
            "architecture_str",
            "PyTorch_MLP",
        )

        signature = infer_signature(
            X_test_np,
            test_preds_fc,
        )

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=(
                REGISTERED_MODEL_NAME
            ),
            signature=signature,
            input_example=X_test_np[:1],
        )

    print(
        "✅ Treino PyTorch concluído!"
    )

    print(
        "🔥 Desempenho OOT: "
        f"MAE={metrics['oot_mae_mw']:.2f} MW | "
        f"nMAE={metrics['oot_nmae_pct']:.2f}% | "
        f"MAE_FC={metrics['oot_mae_fc_pct']:.2f}% | "
        f"R²={metrics['oot_r2_score']:.4f}"
    )

    return (
        run_id,
        metrics["oot_mae_mw"],
    )