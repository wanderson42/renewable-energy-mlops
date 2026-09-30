# src/energy_mlops/models/interfaces.py
from typing import Protocol, TypedDict

import pandas as pd


class OptimizationResult(TypedDict):
    """Resultado produzido por uma execução de otimização."""

    optimization_run_id: str
    best_params: dict


class ModelTrainer(Protocol):
    """Contrato de Interface para Algoritmos de Treinamento."""

    def __call__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        train_file: str,
        test_file: str,
        best_params: dict,
        optimization_run_id: str,
    ) -> tuple[str, float]:
        ...


class ModelOptimizer(Protocol):
    """Contrato de Interface para Algoritmos de Otimização."""

    def __call__(
        self,
        df_train: pd.DataFrame,
        train_file: str,
        n_trials: int = 10,
        n_splits: int = 3,
    ) -> OptimizationResult:
        ...