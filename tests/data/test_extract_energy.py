from unittest.mock import patch

import pandas as pd
import pytest

import energy_mlops.data.extract_energy as energy_module


def _make_ons_dataframe(
    rows: list[dict],
) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "din_instante",
            "nom_estado",
            "nom_tipousina",
            "val_geracao",
        ],
    )


def test_fetch_ons_wind_generation_aggregates_complete_hours():
    """
    Garante que registros de múltiplas usinas sejam
    agregados corretamente por hora.
    """

    df_raw = _make_ons_dataframe(
        [
            {
                "din_instante": "2025-01-01 00:00:00",
                "nom_estado": "BAHIA",
                "nom_tipousina": "EOLIELÉTRICA",
                "val_geracao": "100,5",
            },
            {
                "din_instante": "2025-01-01 00:00:00",
                "nom_estado": "BAHIA",
                "nom_tipousina": "EOLIELÉTRICA",
                "val_geracao": "200.0",
            },
            {
                "din_instante": "2025-01-01 01:00:00",
                "nom_estado": "BAHIA",
                "nom_tipousina": "EOLIELÉTRICA",
                "val_geracao": "150.0",
            },
            {
                "din_instante": "2025-01-01 01:00:00",
                "nom_estado": "BAHIA",
                "nom_tipousina": "EOLIELÉTRICA",
                "val_geracao": "50.0",
            },
        ]
    )

    with patch.object(
        energy_module.pd,
        "read_parquet",
        return_value=df_raw,
    ):
        result = (
            energy_module
            .fetch_ons_wind_generation(
                year=2025,
                month=1,
            )
        )

    assert len(result) == 2

    assert result[
        "wind_generation_mw"
    ].tolist() == [
        300.5,
        200.0,
    ]

    # America/Sao_Paulo -> UTC
    assert (
        result["date"].dt.hour.tolist()
        == [3, 4]
    )


def test_fetch_ons_discards_entire_incomplete_hour():
    """
    Se qualquer registro de uma hora estiver ausente,
    a hora inteira deve ser descartada.

    Isso evita interpretar uma soma parcial como
    geração total da Bahia.
    """

    df_raw = _make_ons_dataframe(
        [
            {
                "din_instante": "2025-01-01 00:00:00",
                "nom_estado": "BAHIA",
                "nom_tipousina": "EOLIELÉTRICA",
                "val_geracao": "100.0",
            },
            {
                "din_instante": "2025-01-01 00:00:00",
                "nom_estado": "BAHIA",
                "nom_tipousina": "EOLIELÉTRICA",
                "val_geracao": "",
            },
            {
                "din_instante": "2025-01-01 01:00:00",
                "nom_estado": "BAHIA",
                "nom_tipousina": "EOLIELÉTRICA",
                "val_geracao": "200.0",
            },
            {
                "din_instante": "2025-01-01 01:00:00",
                "nom_estado": "BAHIA",
                "nom_tipousina": "EOLIELÉTRICA",
                "val_geracao": "50.0",
            },
        ]
    )

    with patch.object(
        energy_module.pd,
        "read_parquet",
        return_value=df_raw,
    ):
        result = (
            energy_module
            .fetch_ons_wind_generation(
                year=2025,
                month=1,
            )
        )

    # 00:00 foi descartada inteiramente.
    assert len(result) == 1

    assert (
        result.loc[
            0,
            "wind_generation_mw",
        ]
        == 250.0
    )

    # 01:00 Brasília -> 04:00 UTC
    assert result.loc[
        0,
        "date",
    ].hour == 4


def test_fetch_ons_rejects_when_all_hours_are_incomplete():
    """
    O extractor deve falhar explicitamente quando
    nenhuma hora completa permanecer.
    """

    df_raw = _make_ons_dataframe(
        [
            {
                "din_instante": "2025-01-01 00:00:00",
                "nom_estado": "BAHIA",
                "nom_tipousina": "EOLIELÉTRICA",
                "val_geracao": "100.0",
            },
            {
                "din_instante": "2025-01-01 00:00:00",
                "nom_estado": "BAHIA",
                "nom_tipousina": "EOLIELÉTRICA",
                "val_geracao": "",
            },
        ]
    )

    with (
        patch.object(
            energy_module.pd,
            "read_parquet",
            return_value=df_raw,
        ),
        pytest.raises(
            ValueError,
            match=(
                "Nenhuma hora completa "
                "permaneceu"
            ),
        ),
    ):
        (
            energy_module
            .fetch_ons_wind_generation(
                year=2025,
                month=1,
            )
        )


def test_fetch_ons_rejects_invalid_month():
    with pytest.raises(
        ValueError,
        match=(
            "month deve estar entre 1 e 12"
        ),
    ):
        (
            energy_module
            .fetch_ons_wind_generation(
                year=2025,
                month=13,
            )
        )