from unittest.mock import MagicMock, patch

import pytest

import energy_mlops.data.extract_capacity as capacity_module


def _make_aneel_response(
    records: list[dict],
    *,
    total: int | None = None,
    success: bool = True,
) -> MagicMock:
    response = MagicMock()

    response.raise_for_status.return_value = (
        None
    )

    if total is None:
        total = len(records)

    response.json.return_value = {
        "success": success,
        "result": {
            "records": records,
            "total": total,
        },
    }

    return response


def _aneel_record(
    record_id: int,
    generation_unit: str,
    capacity: str = "4500.00",
) -> dict:
    return {
        "_id": record_id,
        "CodCEG": (
            "EOL.CV.BA.049382-1.1"
        ),
        "NumUgUsina": generation_unit,
        "SigTipoGeracao": "EOL",
        "SigUFUsina": "BA",
        "MdaPotenciaLiberadaComercial": (
            capacity
        ),
        "DatLiberOpComerRealizado": (
            "12/12/2025"
        ),
    }


def test_fetch_aneel_preserves_distinct_generation_units():
    """
    Duas UGs da mesma usina e mesma data devem
    continuar sendo dois eventos distintos.
    """

    records = [
        _aneel_record(
            record_id=1,
            generation_unit="1",
            capacity="4.500,00",
        ),
        _aneel_record(
            record_id=2,
            generation_unit="2",
            capacity="4500.00",
        ),
    ]

    fake_session = MagicMock()

    fake_session.get.return_value = (
        _make_aneel_response(
            records
        )
    )

    with (
        patch.object(
            capacity_module.requests,
            "Session",
            return_value=MagicMock(),
        ),
        patch.object(
            capacity_module,
            "retry",
            return_value=fake_session,
        ),
    ):
        result = (
            capacity_module
            .fetch_aneel_wind_capacity()
        )

    assert len(result) == 2

    assert result[
        "generation_unit"
    ].tolist() == [
        "1",
        "2",
    ]

    assert result[
        "capacity_added_mw"
    ].tolist() == [
        4.5,
        4.5,
    ]

    assert (
        result["plant_id"]
        .nunique()
        == 1
    )


def test_fetch_aneel_paginates_until_total_is_reached():
    first_page = [
        _aneel_record(
            record_id=1,
            generation_unit="1",
        ),
        _aneel_record(
            record_id=2,
            generation_unit="2",
        ),
    ]

    second_page = [
        _aneel_record(
            record_id=3,
            generation_unit="3",
        ),
    ]

    fake_session = MagicMock()

    fake_session.get.side_effect = [
        _make_aneel_response(
            first_page,
            total=3,
        ),
        _make_aneel_response(
            second_page,
            total=3,
        ),
    ]

    with (
        patch.object(
            capacity_module.requests,
            "Session",
            return_value=MagicMock(),
        ),
        patch.object(
            capacity_module,
            "retry",
            return_value=fake_session,
        ),
    ):
        result = (
            capacity_module
            .fetch_aneel_wind_capacity(
                page_size=2,
            )
        )

    assert len(result) == 3

    assert (
        fake_session.get.call_count
        == 2
    )

    assert result[
        "generation_unit"
    ].tolist() == [
        "1",
        "2",
        "3",
    ]


def test_fetch_aneel_rejects_missing_required_columns():
    incomplete_record = {
        "_id": 1,
        "CodCEG": (
            "EOL.CV.BA.049382-1.1"
        ),
        # NumUgUsina ausente
        "SigTipoGeracao": "EOL",
        "SigUFUsina": "BA",
        "MdaPotenciaLiberadaComercial": (
            "4500.00"
        ),
        "DatLiberOpComerRealizado": (
            "12/12/2025"
        ),
    }

    fake_session = MagicMock()

    fake_session.get.return_value = (
        _make_aneel_response(
            [incomplete_record]
        )
    )

    with (
        patch.object(
            capacity_module.requests,
            "Session",
            return_value=MagicMock(),
        ),
        patch.object(
            capacity_module,
            "retry",
            return_value=fake_session,
        ),
        pytest.raises(
            ValueError,
            match=(
                "Resposta ANEEL sem "
                "colunas obrigatórias"
            ),
        ),
    ):
        (
            capacity_module
            .fetch_aneel_wind_capacity()
        )


def test_fetch_aneel_rejects_invalid_page_size():
    with pytest.raises(
        ValueError,
        match=(
            "page_size deve ser "
            "maior que zero"
        ),
    ):
        (
            capacity_module
            .fetch_aneel_wind_capacity(
                page_size=0,
            )
        )