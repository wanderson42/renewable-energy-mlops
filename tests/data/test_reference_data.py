import pandas as pd
import pytest
from pandera.errors import SchemaError

from energy_mlops.data.reference_data import (
    load_bahia_wind_capacity_checkpoints,
)


def _valid_capacity_rows() -> list[dict]:
    return [
        {
            "edition": 35,
            "publication_date": "2024-10-02",
            "reference_date": "2024-10-01",
            "available_from": "2024-10-02",
            "capacity_mw": 10403.3,
            "state": "BA",
            "source": "ABEEOLICA_INFOVENTO",
            "reported_reference_period": "2024-10",
            "date_basis": "explicit_source_reference",
            "source_url": (
                "https://example.com/infovento35.pdf"
            ),
            "notes": "Checkpoint de teste.",
        },
        {
            "edition": 34,
            "publication_date": "2024-03-21",
            "reference_date": "2023-08-01",
            "available_from": "2024-03-21",
            "capacity_mw": 9715.9,
            "state": "BA",
            "source": "ABEEOLICA_INFOVENTO",
            "reported_reference_period": "2023-08",
            "date_basis": "explicit_source_reference",
            "source_url": (
                "https://example.com/infovento34.pdf"
            ),
            "notes": "Checkpoint de teste.",
        },
    ]


def _write_capacity_csv(
    tmp_path,
    rows: list[dict],
):
    path = tmp_path / "capacity_checkpoints.csv"

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )

    return path


def test_load_capacity_checkpoints_parses_dates_and_sorts(
    tmp_path,
):
    path = _write_capacity_csv(
        tmp_path,
        _valid_capacity_rows(),
    )

    result = load_bahia_wind_capacity_checkpoints(
        path
    )

    assert result["edition"].tolist() == [
        34,
        35,
    ]

    assert pd.api.types.is_datetime64_any_dtype(
        result["publication_date"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result["reference_date"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result["available_from"]
    )

    assert result[
        "available_from"
    ].is_monotonic_increasing


def test_load_capacity_checkpoints_rejects_missing_column(
    tmp_path,
):
    rows = _valid_capacity_rows()

    for row in rows:
        row.pop("capacity_mw")

    path = _write_capacity_csv(
        tmp_path,
        rows,
    )

    with pytest.raises(SchemaError):
        load_bahia_wind_capacity_checkpoints(
            path
        )


def test_load_capacity_checkpoints_rejects_reference_after_availability(
    tmp_path,
):
    rows = _valid_capacity_rows()

    rows[0]["reference_date"] = "2024-10-03"
    rows[0]["available_from"] = "2024-10-02"

    path = _write_capacity_csv(
        tmp_path,
        rows,
    )

    with pytest.raises(SchemaError):
        load_bahia_wind_capacity_checkpoints(
            path
        )


def test_load_capacity_checkpoints_rejects_availability_before_publication(
    tmp_path,
):
    rows = _valid_capacity_rows()

    rows[0]["publication_date"] = "2024-10-02"
    rows[0]["available_from"] = "2024-10-01"

    path = _write_capacity_csv(
        tmp_path,
        rows,
    )

    with pytest.raises(SchemaError):
        load_bahia_wind_capacity_checkpoints(
            path
        )


def test_load_capacity_checkpoints_rejects_unknown_file(
    tmp_path,
):
    path = (
        tmp_path
        / "arquivo_inexistente.csv"
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Arquivo de checkpoints "
            "de capacidade não encontrado"
        ),
    ):
        load_bahia_wind_capacity_checkpoints(
            path
        )


def test_default_capacity_reference_file_is_valid():
    result = (
        load_bahia_wind_capacity_checkpoints()
    )

    assert not result.empty

    assert (
        result["state"]
        == "BA"
    ).all()

    assert (
        result["source"]
        == "ABEEOLICA_INFOVENTO"
    ).all()

    assert result[
        "available_from"
    ].is_monotonic_increasing

    assert (
        result["reference_date"]
        <= result["available_from"]
    ).all()

    assert (
        result["available_from"]
        == result["publication_date"]
    ).all()