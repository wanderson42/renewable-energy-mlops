import pandas as pd

from energy_mlops.data.schema import ModelFeatureSchema


def get_model_feature_columns() -> list[str]:
    return list(ModelFeatureSchema.to_schema().columns)


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = get_model_feature_columns()

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Features obrigatórias ausentes no DataFrame: {missing}"
        )

    X = df[feature_columns].copy()

    return ModelFeatureSchema.validate(X)