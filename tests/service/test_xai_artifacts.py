import json

from energy_mlops.service.xai_artifacts import (
    load_xai_artifacts_from_mlflow,
)


def test_load_xai_artifacts_uses_served_model_run_id(
    mocker,
    tmp_path,
):
    """
    Garante que os artefatos XAI são carregados
    diretamente pela Run informada pela FastAPI.

    O helper não deve descobrir novamente o
    @champion no Model Registry.
    """

    run_id = "fake-run-id-123"

    # --------------------------------------------------
    # Diretório falso contendo os gráficos SHAP
    # --------------------------------------------------
    explainability_dir = (
        tmp_path / "explainability"
    )

    explainability_dir.mkdir()

    # --------------------------------------------------
    # model_summary.json falso
    # --------------------------------------------------
    summary_path = (
        tmp_path / "model_summary.json"
    )

    expected_summary = {
        "model_name": (
            "ensemble_lgb_xgb_rf_bahia"
        ),
        "architecture": (
            "TemporalStackingRegressor"
        ),
        "meta_learner": (
            "LinearRegression"
        ),
        "meta_weights": {
            "lgbm": 0.2,
            "xgboost": 0.3,
            "rf": 0.5,
        },
    }

    summary_path.write_text(
        json.dumps(expected_summary),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Mock do MLflow Artifact Store
    #
    # Primeira chamada:
    # runs:/<run_id>/explainability
    #
    # Segunda chamada:
    # runs:/<run_id>/model_summary.json
    # --------------------------------------------------
    mock_download = mocker.patch(
        (
            "energy_mlops.service.xai_artifacts."
            "mlflow.artifacts.download_artifacts"
        ),
        side_effect=[
            str(explainability_dir),
            str(summary_path),
        ],
    )

    (
        result_dir,
        model_summary,
        error,
    ) = load_xai_artifacts_from_mlflow(
        run_id
    )

    # --------------------------------------------------
    # Resultado
    # --------------------------------------------------
    assert result_dir == str(
        explainability_dir
    )

    assert model_summary == (
        expected_summary
    )

    assert error is None

    # --------------------------------------------------
    # O ponto arquitetural principal:
    #
    # os dois artefatos devem pertencer
    # exatamente à Run que a API informou.
    # --------------------------------------------------
    assert (
        mock_download.call_count
        == 2
    )

    mock_download.assert_any_call(
        artifact_uri=(
            "runs:/fake-run-id-123/"
            "explainability"
        )
    )

    mock_download.assert_any_call(
        artifact_uri=(
            "runs:/fake-run-id-123/"
            "model_summary.json"
        )
    )


def test_load_xai_artifacts_allows_missing_model_summary(
    mocker,
    tmp_path,
):
    """
    A ausência de model_summary.json não deve
    impedir a exibição dos gráficos SHAP.
    """

    explainability_dir = (
        tmp_path / "explainability"
    )

    explainability_dir.mkdir()

    mock_download = mocker.patch(
        (
            "energy_mlops.service.xai_artifacts."
            "mlflow.artifacts.download_artifacts"
        ),
        side_effect=[
            str(explainability_dir),
            Exception(
                "model_summary.json not found"
            ),
        ],
    )

    (
        result_dir,
        model_summary,
        error,
    ) = load_xai_artifacts_from_mlflow(
        "fake-run-id-123"
    )

    assert result_dir == str(
        explainability_dir
    )

    assert model_summary is None

    # A falta do resumo não é uma falha
    # do XAI como um todo.
    assert error is None

    assert (
        mock_download.call_count
        == 2
    )


def test_load_xai_artifacts_returns_error_when_shap_is_unavailable(
    mocker,
):
    """
    Se a própria pasta explainability não puder
    ser carregada, o helper deve devolver erro.
    """

    mock_download = mocker.patch(
        (
            "energy_mlops.service.xai_artifacts."
            "mlflow.artifacts.download_artifacts"
        ),
        side_effect=Exception(
            "Artifact Store unavailable"
        ),
    )

    (
        result_dir,
        model_summary,
        error,
    ) = load_xai_artifacts_from_mlflow(
        "fake-run-id-123"
    )

    assert result_dir is None
    assert model_summary is None

    assert (
        "Artifact Store unavailable"
        in error
    )

    mock_download.assert_called_once_with(
        artifact_uri=(
            "runs:/fake-run-id-123/"
            "explainability"
        )
    )


def test_load_xai_artifacts_rejects_empty_run_id(
    mocker,
):
    """
    Sem run_id não existe identidade suficiente
    para garantir que o XAI pertence ao modelo
    efetivamente servido.
    """

    mock_download = mocker.patch(
            "energy_mlops.service.xai_artifacts."
            "mlflow.artifacts.download_artifacts"
    )

    (
        result_dir,
        model_summary,
        error,
    ) = load_xai_artifacts_from_mlflow(
        ""
    )

    assert result_dir is None
    assert model_summary is None

    assert (
        "Run ID"
        in error
    )

    mock_download.assert_not_called()