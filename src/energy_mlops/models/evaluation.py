from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


@dataclass(frozen=True)
class ModelEvaluation:
    """Métricas operacionais calculadas sobre um único OOT."""

    mae_mw: float
    nmae_pct: float


def _normalize_feature_names(
    feature_names: Any,
) -> list[str] | None:
    """Normaliza uma coleção de nomes de features para ``list[str]``."""

    if feature_names is None:
        return None

    names = [
        str(name)
        for name in feature_names
    ]

    if not names:
        return None

    return names


def _extract_estimator_feature_names(
    estimator: Any,
) -> list[str] | None:
    """Extrai a ordem de features conhecida por um estimator."""

    feature_names = _normalize_feature_names(
        getattr(
            estimator,
            "feature_names_in_",
            None,
        )
    )

    if feature_names is not None:
        return feature_names

    get_booster = getattr(
        estimator,
        "get_booster",
        None,
    )

    if callable(get_booster):
        booster = get_booster()

        feature_names = _normalize_feature_names(
            getattr(
                booster,
                "feature_names",
                None,
            )
        )

        if feature_names is not None:
            return feature_names

    return None


def get_model_feature_order(
    model: Any,
) -> list[str]:
    """
    Recupera a ordem de features utilizada pelo modelo carregado.

    A procura cobre o estimator principal e estruturas compostas usadas
    por versões históricas do ensemble.
    """

    feature_names = _extract_estimator_feature_names(
        model
    )

    if feature_names is not None:
        return feature_names

    named_estimators = getattr(
        model,
        "named_estimators_",
        None,
    )

    if named_estimators is not None:
        for estimator in named_estimators.values():
            feature_names = (
                _extract_estimator_feature_names(
                    estimator
                )
            )

            if feature_names is not None:
                return feature_names

    estimators = getattr(
        model,
        "estimators_",
        None,
    )

    if estimators is not None:
        for estimator in estimators:
            feature_names = (
                _extract_estimator_feature_names(
                    estimator
                )
            )

            if feature_names is not None:
                return feature_names

    raise RuntimeError(
        "Não foi possível determinar a ordem "
        "de features esperada pelo modelo."
    )


def align_features_to_model(
    model: Any,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Reordena as features conforme o contrato do modelo carregado."""

    expected_features = get_model_feature_order(
        model
    )

    missing_features = [
        feature
        for feature in expected_features
        if feature not in X.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Features obrigatórias do modelo "
            "ausentes no dataset de avaliação: "
            f"{missing_features}"
        )

    return (
        X.loc[
            :,
            expected_features,
        ]
        .copy()
    )


def evaluate_model_on_oot(
    model: Any,
    X: pd.DataFrame,
    df_oot: pd.DataFrame,
) -> ModelEvaluation:
    """
    Avalia um modelo sobre um OOT já observado.

    O mesmo contrato é usado pelo monitoring e pelo Quality Gate para
    impedir que Champion e Challenger sejam comparados em períodos
    diferentes ou com regras de cálculo diferentes.
    """

    required_columns = {
        "wind_generation_mw",
        "capacidade_mw",
    }

    missing_columns = required_columns.difference(
        df_oot.columns
    )

    if missing_columns:
        raise ValueError(
            "Avaliação OOT exige ground truth e capacidade. "
            "Colunas ausentes: "
            f"{sorted(missing_columns)}"
        )

    if len(X) != len(df_oot):
        raise ValueError(
            "Features e OOT devem possuir o mesmo número "
            "de observações."
        )

    X_model = align_features_to_model(
        model,
        X,
    )

    predictions_fc = (
        np.asarray(
            model.predict(X_model),
            dtype=float,
        )
        .reshape(-1)
    )

    capacidade_mw = (
        df_oot["capacidade_mw"]
        .to_numpy(dtype=float)
    )

    actual_mw = (
        df_oot["wind_generation_mw"]
        .to_numpy(dtype=float)
    )

    if len(predictions_fc) != len(actual_mw):
        raise ValueError(
            "Número de previsões diferente do número "
            "de observações OOT."
        )

    max_capacity_mw = float(
        np.max(capacidade_mw)
    )

    if (
        not np.isfinite(max_capacity_mw)
        or max_capacity_mw <= 0
    ):
        raise ValueError(
            "Capacidade máxima inválida para "
            "o cálculo do nMAE: "
            f"{max_capacity_mw}."
        )

    predictions_mw = (
        predictions_fc
        * capacidade_mw
    )

    mae_mw = float(
        mean_absolute_error(
            actual_mw,
            predictions_mw,
        )
    )

    nmae_pct = (
        mae_mw
        / max_capacity_mw
    ) * 100

    return ModelEvaluation(
        mae_mw=mae_mw,
        nmae_pct=float(nmae_pct),
    )
