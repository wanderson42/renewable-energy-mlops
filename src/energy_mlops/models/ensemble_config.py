from collections.abc import Sequence

# ============================================================
# Arquitetura padrão
# ============================================================

DEFAULT_BASE_ESTIMATORS: tuple[str, ...] = (
    "lgbm",
    "xgboost",
    "rf",
)


# ============================================================
# Estimadores suportados pelo pipeline
# ============================================================

SUPPORTED_BASE_ESTIMATORS = frozenset(
    DEFAULT_BASE_ESTIMATORS
)


# ============================================================
# Nome do grupo de parâmetros produzido pelo optimizer
#
# Exemplo:
#
# {
#     "lgb_params": {...},
#     "xgb_params": {...},
#     "rf_params": {...},
# }
# ============================================================

PARAM_GROUP_BY_ESTIMATOR = {
    "lgbm": "lgb_params",
    "xgboost": "xgb_params",
    "rf": "rf_params",
}


# ============================================================
# Prefixo usado nos parâmetros do Optuna / MLflow
#
# Exemplos:
#
# lgb_n_estimators
# xgb_learning_rate
# rf_max_depth
# ============================================================

PARAM_PREFIX_BY_ESTIMATOR = {
    "lgbm": "lgb",
    "xgboost": "xgb",
    "rf": "rf",
}


# ============================================================
# Labels curtos para nomes de Runs no MLflow
#
# Exemplos:
#
# Stacking_LGB_XGB_RF_Bahia
# Stacking_LGB_RF_Bahia
# ============================================================

ESTIMATOR_LABEL_BY_NAME = {
    "lgbm": "LGB",
    "xgboost": "XGB",
    "rf": "RF",
}


# ============================================================
# Validação do contrato de arquitetura
# ============================================================

def validate_enabled_estimators(
    enabled_estimators: Sequence[str],
) -> tuple[str, ...]:
    """
    Valida a composição dos estimadores base do Temporal Stacking.

    Regras atuais do pipeline:

    - pelo menos dois estimadores base;
    - nenhum estimador duplicado;
    - somente estimadores suportados;
    - LightGBM é obrigatório enquanto o pipeline de SHAP
      estiver ancorado no estimador 'lgbm'.

    Retorna uma tupla validada para preservar uma ordem
    determinística dos estimadores.
    """

    estimators = tuple(enabled_estimators)

    if len(estimators) < 2:
        raise ValueError(
            "Temporal Stacking requer pelo menos "
            "dois estimadores base."
        )

    if len(set(estimators)) != len(estimators):
        raise ValueError(
            "enabled_estimators contém estimadores "
            "duplicados."
        )

    unknown_estimators = (
        set(estimators)
        - SUPPORTED_BASE_ESTIMATORS
    )

    if unknown_estimators:
        raise ValueError(
            "Estimadores não suportados: "
            f"{sorted(unknown_estimators)}"
        )

    # O pipeline atual de explicabilidade usa:
    #
    # ensemble.named_estimators_["lgbm"]
    #
    # Portanto, enquanto SHAP estiver ancorado no LightGBM,
    # não permitimos uma arquitetura sem esse estimador.
    if "lgbm" not in estimators:
        raise ValueError(
            "O pipeline atual requer 'lgbm' porque "
            "a explicabilidade SHAP está ancorada "
            "nesse estimador."
        )

    return estimators