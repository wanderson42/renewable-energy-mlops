# src/energy_mlops/models/interfaces.py

from typing import Protocol, TypedDict

import pandas as pd


class OptimizationResult(TypedDict):
    """Resultado produzido por uma execução de otimização."""

    optimization_run_id: str
    best_params: dict


class ModelTrainer(Protocol):
    """
    Contrato de interface para algoritmos de treinamento.

    Um trainer pode consumir ou não uma etapa de otimização anterior.
    Quando não houver optimizer, ``best_params`` e
    ``optimization_run_id`` serão ``None``.
    """

    def __call__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        train_file: str,
        test_file: str,
        best_params: dict | None = None,
        optimization_run_id: str | None = None,
    ) -> tuple[str, float]:
        ...


class ModelOptimizer(Protocol):
    """Contrato de interface para algoritmos de otimização."""

    def __call__(
        self,
        df_train: pd.DataFrame,
        train_file: str,
        n_trials: int = 10,
        n_splits: int = 3,
    ) -> OptimizationResult:
        ...