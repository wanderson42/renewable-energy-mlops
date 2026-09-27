# src/energy_mlops/models/interfaces.py
from typing import Protocol

import pandas as pd


class ModelTrainer(Protocol):
    """Contrato de Interface para Algoritmos de Treinamento."""
    def __call__(
        self, 
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame, 
        train_file: str, 
        test_file: str
    ) -> tuple[str, float]:
        ...

class ModelOptimizer(Protocol):
    """Contrato de Interface para Algoritmos de Otimização de Hiperparâmetros."""
    def __call__(
        self, 
        df_train: pd.DataFrame, 
        train_file: str, 
        n_trials: int = 10, 
        n_splits: int = 3
    ) -> dict:
        ...