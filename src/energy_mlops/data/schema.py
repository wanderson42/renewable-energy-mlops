import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class WeatherSchema(pa.DataFrameModel):
    """Contrato de dados para as Features climáticas."""

    date: Series[pd.Timestamp] = pa.Field(nullable=False)
    temperature_2m: Series[float] = pa.Field(
        ge=-50.0,
        le=60.0,
        nullable=False,
    )
    wind_speed_100m: Series[float] = pa.Field(
        ge=0.0,
        le=200.0,
        nullable=False,
    )
    wind_direction_100m: Series[float] = pa.Field(
        ge=0.0,
        le=360.0,
        nullable=False,
    )

    class Config:
        coerce = True
        strict = True


class EnergySchema(pa.DataFrameModel):
    """Contrato de dados para a geração de energia do ONS."""

    date: Series[pd.Timestamp] = pa.Field(nullable=False)
    wind_generation_mw: Series[float] = pa.Field(
        ge=0.0,
        le=30000.0,
        nullable=False,
    )

    class Config:
        coerce = True
        strict = True


class ModelFeatureSchema(pa.DataFrameModel):
    """Contrato das features finais utilizadas pelo modelo."""

    temperature_2m: Series[float] = pa.Field(nullable=False)
    wind_speed_100m: Series[float] = pa.Field(
        ge=0.0,
        le=200.0,
        nullable=False,
    )
    wind_direction_100m: Series[float] = pa.Field(
        ge=0.0,
        le=360.0,
        nullable=False,
    )

    wind_temp_ratio: Series[float] = pa.Field(nullable=False)

    hour_sin: Series[float] = pa.Field(nullable=False)
    hour_cos: Series[float] = pa.Field(nullable=False)
    month_sin: Series[float] = pa.Field(nullable=False)
    month_cos: Series[float] = pa.Field(nullable=False)

    wind_speed_roll_mean_3h: Series[float] = pa.Field(nullable=False)

    class Config:
        coerce = True
        strict = True


class CapacityEventSchema(pa.DataFrameModel):
    """
    Contrato dos eventos de liberação comercial
    utilizados para reconciliação com a ANEEL.
    """

    effective_date: Series[pd.Timestamp] = pa.Field(
        nullable=False,
    )

    plant_id: Series[str] = pa.Field(
        nullable=False,
    )

    generation_unit: Series[str] = pa.Field(
        nullable=False,
    )

    state: Series[str] = pa.Field(
        nullable=False,
    )

    generation_type: Series[str] = pa.Field(
        nullable=False,
    )

    capacity_added_mw: Series[float] = pa.Field(
        gt=0.0,
        nullable=False,
    )

    class Config:
        coerce = True
        strict = True


class CapacityCheckpointSchema(pa.DataFrameModel):
    """
    Contrato dos checkpoints de capacidade eólica da Bahia
    reportados nos boletins INFOVENTO da ABEEólica.

    reference_date:
        Data/período histórico ao qual o valor é atribuído.

    publication_date:
        Data de publicação do boletim.

    available_from:
        Data a partir da qual o pipeline pode utilizar
        causalmente o checkpoint.

    date_basis:
        Informa como reference_date foi determinada.
    """

    edition: Series[int] = pa.Field(
        ge=1,
        nullable=False,
        unique=True,
    )

    publication_date: Series[pd.Timestamp] = pa.Field(
        nullable=False,
    )

    reference_date: Series[pd.Timestamp] = pa.Field(
        nullable=False,
        unique=True,
    )

    available_from: Series[pd.Timestamp] = pa.Field(
        nullable=False,
        unique=True,
    )

    capacity_mw: Series[float] = pa.Field(
        gt=0.0,
        nullable=False,
    )

    state: Series[str] = pa.Field(
        isin=["BA"],
        nullable=False,
    )

    source: Series[str] = pa.Field(
        isin=["ABEEOLICA_INFOVENTO"],
        nullable=False,
    )

    reported_reference_period: Series[str] = pa.Field(
        str_matches=r"^\d{4}-\d{2}$",
        nullable=False,
    )

    date_basis: Series[str] = pa.Field(
        isin=[
            "explicit_source_reference",
            "publication_date_proxy",
        ],
        nullable=False,
    )

    source_url: Series[str] = pa.Field(
        str_startswith="https://",
        nullable=False,
    )

    notes: Series[str] = pa.Field(
        nullable=False,
    )

    @pa.dataframe_check
    def reference_not_after_availability(
        cls,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Um checkpoint não pode se referir a uma data
        posterior ao momento em que ficou disponível.
        """
        return (
            df["reference_date"]
            <= df["available_from"]
        )

    @pa.dataframe_check
    def availability_matches_publication(
        cls,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Política causal atual do projeto:
        o checkpoint fica disponível na publicação.
        """
        return (
            df["available_from"]
            == df["publication_date"]
        )

    class Config:
        coerce = True
        strict = True