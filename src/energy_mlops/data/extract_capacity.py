import json
import logging

import pandas as pd
import requests
from retry_requests import retry

from energy_mlops.data.schema import (
    CapacityEventSchema,
)

logger = logging.getLogger(__name__)


ANEEL_DATASTORE_URL = (
    "https://dadosabertos.aneel.gov.br/"
    "api/action/datastore_search"
)

ANEEL_COMMERCIAL_OPERATION_RESOURCE_ID = (
    "75419902-c692-498b-a6ef-85f6d4beb5b2"
)

DEFAULT_STATE = "BA"
DEFAULT_GENERATION_TYPE = "EOL"

KW_PER_MW = 1000.0


def _parse_numeric_column(
    series: pd.Series,
) -> pd.Series:
    """
    Converte valores numéricos retornados pela ANEEL.

    Suporta tanto:

        4200.00

    quanto:

        4.200,00
    """

    values = (
        series.astype("string")
        .str.strip()
    )

    has_decimal_comma = (
        values.str.contains(
            ",",
            na=False,
        )
    )

    values.loc[
        has_decimal_comma
    ] = (
        values.loc[
            has_decimal_comma
        ]
        .str.replace(
            ".",
            "",
            regex=False,
        )
        .str.replace(
            ",",
            ".",
            regex=False,
        )
    )

    return pd.to_numeric(
        values,
        errors="coerce",
    )


def fetch_aneel_wind_capacity(
    state: str = DEFAULT_STATE,
    generation_type: str = DEFAULT_GENERATION_TYPE,
    api_token: str | None = None,
    page_size: int = 1000,
) -> pd.DataFrame:
    """
    Extrai eventos de liberação comercial de unidades
    geradoras eólicas da ANEEL.

    Esta fonte é utilizada para reconciliação e
    diagnóstico da evolução da capacidade eólica.

    Ela NÃO representa a fonte canônica de
    `capacidade_mw` utilizada pelo modelo. Essa
    responsabilidade pertence aos checkpoints
    INFOVENTO carregados por `reference_data.py`.

    Defaults do projeto:

        state="BA"
        generation_type="EOL"
    """

    if page_size <= 0:
        raise ValueError(
            "page_size deve ser maior que zero."
        )

    state = state.strip().upper()
    generation_type = (
        generation_type
        .strip()
        .upper()
    )

    logger.info(
        "Consultando eventos comerciais ANEEL "
        "(state=%s, generation_type=%s).",
        state,
        generation_type,
    )

    session = retry(
        requests.Session(),
        retries=5,
        backoff_factor=0.5,
    )

    headers = {}

    if api_token:
        headers["Authorization"] = (
            api_token
        )

    filters = {
        "SigUFUsina": state,
        "SigTipoGeracao": (
            generation_type
        ),
    }

    records: list[dict] = []

    offset = 0
    total: int | None = None

    try:
        while (
            total is None
            or offset < total
        ):
            params = {
                "resource_id": (
                    ANEEL_COMMERCIAL_OPERATION_RESOURCE_ID
                ),
                "limit": page_size,
                "offset": offset,
                "filters": json.dumps(
                    filters,
                    ensure_ascii=False,
                ),
            }

            response = session.get(
                ANEEL_DATASTORE_URL,
                params=params,
                headers=headers,
                timeout=30,
            )

            response.raise_for_status()

            payload = response.json()

            if not payload.get(
                "success"
            ):
                raise RuntimeError(
                    "A API da ANEEL retornou "
                    "success=False."
                )

            result = payload.get(
                "result",
                {},
            )

            page_records = (
                result.get(
                    "records",
                    [],
                )
            )

            total = int(
                result.get(
                    "total",
                    0,
                )
            )

            records.extend(
                page_records
            )

            logger.info(
                "ANEEL: %s/%s registros recebidos.",
                len(records),
                total,
            )

            if not page_records:
                break

            offset += len(
                page_records
            )

    except Exception:
        logger.exception(
            "Falha durante a extração "
            "de eventos da ANEEL."
        )
        raise

    if not records:
        raise ValueError(
            "A consulta ANEEL não retornou eventos "
            f"para state={state!r} e "
            f"generation_type={generation_type!r}."
        )

    df_raw = pd.DataFrame(
        records
    )

    required_columns = {
        "CodCEG",
        "NumUgUsina",
        "SigTipoGeracao",
        "SigUFUsina",
        "MdaPotenciaLiberadaComercial",
        "DatLiberOpComerRealizado",
    }

    missing_columns = (
        required_columns
        - set(df_raw.columns)
    )

    if missing_columns:
        raise ValueError(
            "Resposta ANEEL sem colunas "
            "obrigatórias: "
            f"{sorted(missing_columns)}"
        )

    # O CKAN utiliza _id como identificador
    # interno do registro. Caso esteja presente,
    # remove apenas duplicações reais da resposta.
    if "_id" in df_raw.columns:
        df_raw = (
            df_raw
            .drop_duplicates(
                subset=["_id"]
            )
            .copy()
        )

    effective_date = pd.to_datetime(
        df_raw[
            "DatLiberOpComerRealizado"
        ],
        errors="coerce",
        dayfirst=True,
    )

    capacity_kw = (
        _parse_numeric_column(
            df_raw[
                "MdaPotenciaLiberadaComercial"
            ]
        )
    )

    df_events = pd.DataFrame(
        {
            "effective_date": (
                effective_date
            ),
            "plant_id": (
                df_raw["CodCEG"]
                .astype("string")
                .str.strip()
            ),
            "generation_unit": (
                df_raw[
                    "NumUgUsina"
                ]
                .astype("string")
                .str.strip()
            ),
            "state": (
                df_raw[
                    "SigUFUsina"
                ]
                .astype("string")
                .str.strip()
                .str.upper()
            ),
            "generation_type": (
                df_raw[
                    "SigTipoGeracao"
                ]
                .astype("string")
                .str.strip()
                .str.upper()
            ),
            "capacity_added_mw": (
                capacity_kw
                / KW_PER_MW
            ),
        }
    )

    invalid_mask = (
        df_events[
            "effective_date"
        ].isna()
        | df_events[
            "plant_id"
        ].isna()
        | df_events[
            "generation_unit"
        ].isna()
        | df_events[
            "capacity_added_mw"
        ].isna()
        | (
            df_events[
                "capacity_added_mw"
            ]
            <= 0.0
        )
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    if invalid_count:
        logger.warning(
            "ANEEL: descartando %s registros "
            "sem data comercial ou potência "
            "válida.",
            invalid_count,
        )

    df_events = (
        df_events.loc[
            ~invalid_mask
        ]
        .copy()
    )

    if df_events.empty:
        raise ValueError(
            "Nenhum evento comercial ANEEL "
            "válido permaneceu após a "
            "normalização."
        )

    df_events.sort_values(
        [
            "effective_date",
            "plant_id",
            "generation_unit",
        ],
        inplace=True,
    )

    df_events.reset_index(
        drop=True,
        inplace=True,
    )

    validated = (
        CapacityEventSchema.validate(
            df_events
        )
    )

    logger.info(
        "Extração ANEEL concluída: "
        "%s eventos comerciais válidos.",
        len(validated),
    )

    return validated


if __name__ == "__main__":
    events = (
        fetch_aneel_wind_capacity()
    )

    print(
        "\nEventos ANEEL:\n"
    )

    print(
        events.tail()
    )

    print(
        "\nTotal de eventos válidos:",
        len(events),
    )

    print(
        "Capacidade somada nos eventos:",
        round(
            events[
                "capacity_added_mw"
            ].sum(),
            3,
        ),
        "MW",
    )