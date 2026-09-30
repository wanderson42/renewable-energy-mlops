import json

import mlflow


def load_xai_artifacts_from_mlflow(
    run_id: str,
) -> tuple[str | None, dict | None, str | None]:
    """
    Carrega os artefatos de explicabilidade associados
    exatamente à Run servida pela FastAPI.

    A identidade operacional do Champion não é
    descoberta aqui. O run_id deve vir de /model-info.

    Returns
    -------
    tuple
        (
            local_explainability_dir,
            model_summary,
            error_message,
        )
    """

    if not run_id:
        return (
            None,
            None,
            "Run ID do modelo Champion não foi informado.",
        )

    try:
        shap_artifact_uri = (
            f"runs:/{run_id}/explainability"
        )

        summary_artifact_uri = (
            f"runs:/{run_id}/model_summary.json"
        )

        local_explainability_dir = (
            mlflow.artifacts.download_artifacts(
                artifact_uri=shap_artifact_uri
            )
        )

        model_summary = None

        try:
            local_summary_path = (
                mlflow.artifacts.download_artifacts(
                    artifact_uri=summary_artifact_uri
                )
            )

            with open(
                local_summary_path,
                "r",
                encoding="utf-8",
            ) as file:
                model_summary = json.load(
                    file
                )

        except Exception:  # noqa: BLE001
            model_summary = None

        return (
            local_explainability_dir,
            model_summary,
            None,
        )

    except Exception as e:  # noqa: BLE001
        return (
            None,
            None,
            str(e),
        )