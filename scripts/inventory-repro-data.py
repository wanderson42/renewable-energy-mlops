"""Read-only preflight, run via stdin inside each MLflow container."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version

MODEL_NAME = "ensemble_lgb_xgb_rf_bahia"
EXPECTED_VERSION = 17
EXPECTED_RUN_ID = "1d13a61244c54f06aa70f43a9993ea37"
BUCKETS = ("energy-lake", "mlflow-artifacts")


def inventory(role, mlflow_client, s3_client):
    """Query metadata only; never download, upload or modify resources."""
    registered = []
    token = None
    while True:
        page = mlflow_client.search_registered_models(max_results=100, page_token=token)
        registered.extend(model.name for model in page)
        token = page.token
        if not token:
            break

    result = {
        "role": role,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "registered_models": sorted(registered),
        "model": None,
        "buckets": {},
    }
    if role == "source":
        model = mlflow_client.get_model_version_by_alias(MODEL_NAME, "champion")
        if int(model.version) != EXPECTED_VERSION or model.run_id != EXPECTED_RUN_ID:
            raise RuntimeError("Source champion differs from the v17 baseline; review before taking a snapshot.")
        run = mlflow_client.get_run(model.run_id)
        result["model"] = {
            "name": MODEL_NAME,
            "alias": "champion",
            "version": int(model.version),
            "run_id": model.run_id,
            "source": model.source,
            "download_uri": mlflow_client.get_model_version_download_uri(MODEL_NAME, model.version),
            "run_artifact_uri": run.info.artifact_uri,
        }
    elif registered:
        raise RuntimeError("Destination Registry is not empty; review before planning a restore.")

    present = {bucket["Name"] for bucket in s3_client.list_buckets()["Buckets"]}
    result["additional_buckets"] = sorted(present.difference(BUCKETS))
    for bucket in BUCKETS:
        count = total_bytes = 0
        if bucket in present:
            for page in s3_client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
                for item in page.get("Contents", []):
                    count += 1
                    total_bytes += item["Size"]
        elif role == "source":
            raise RuntimeError(f"Expected source bucket is missing: {bucket}")
        result["buckets"][bucket] = {
            "present": bucket in present,
            "objects": count,
            "bytes": total_bytes,
        }
        if role == "destination" and count:
            raise RuntimeError(f"Destination bucket is not empty: {bucket}; review before planning a restore.")
    if role == "destination" and result["additional_buckets"]:
        raise RuntimeError("Destination has unexpected buckets; review before planning a restore.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("source", "destination"))
    args = parser.parse_args()
    # These packages belong to the server image, not the project's Poetry env.
    import boto3
    from botocore.config import Config
    from mlflow import MlflowClient

    endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    if endpoint != "http://rustfs:9000":
        raise RuntimeError("Expected the cluster-internal RustFS endpoint http://rustfs:9000.")
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "15"
    client = MlflowClient(tracking_uri="http://127.0.0.1:5000", registry_uri="http://127.0.0.1:5000")
    s3 = boto3.client("s3", endpoint_url=endpoint, config=Config(
        signature_version="s3v4", s3={"addressing_style": "path"},
        connect_timeout=5, read_timeout=15, retries={"max_attempts": 2},
    ))
    result = inventory(args.role, client, s3)
    result["runtime"] = {
        "python": sys.version.split()[0],
        "packages": {name: version(name) for name in (
            "mlflow", "boto3", "botocore", "psycopg2-binary", "s3transfer",
            "jmespath", "urllib3", "python-dateutil", "six",
        )},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
