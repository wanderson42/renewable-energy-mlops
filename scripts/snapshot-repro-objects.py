"""Inventory/export inside MLflow; verify the snapshot locally using stdlib."""

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

BUCKETS = ("energy-lake", "mlflow-artifacts")
MODEL_NAME = "ensemble_lgb_xgb_rf_bahia"
RUN_ID = "1d13a61244c54f06aa70f43a9993ea37"


def model_identity(client):
    model = client.get_model_version_by_alias(MODEL_NAME, "champion")
    if int(model.version) != 17 or model.run_id != RUN_ID:
        raise RuntimeError("Champion identity differs from the v17 baseline.")
    return {
        "name": MODEL_NAME, "version": 17, "run_id": RUN_ID,
        "source": model.source,
        "download_uri": client.get_model_version_download_uri(MODEL_NAME, model.version),
    }


def object_inventory(s3):
    present = {item["Name"] for item in s3.list_buckets()["Buckets"]}
    if not set(BUCKETS).issubset(present):
        raise RuntimeError("One of the required source buckets is missing.")
    records = []
    for bucket in BUCKETS:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
            for item in page.get("Contents", []):
                records.append({
                    "bucket": bucket, "key": item["Key"], "size": item["Size"],
                    "etag": item["ETag"], "last_modified": item["LastModified"].isoformat(),
                })
    records.sort(key=lambda item: (item["bucket"], item["key"]))
    for index, record in enumerate(records):
        # Object keys never become filesystem/archive paths.
        record["entry"] = f"objects/{index:08d}"
    return records


def validate_manifest(manifest):
    model = manifest["model"]
    if manifest["format_version"] != 1 or tuple(manifest["buckets"]) != BUCKETS:
        raise RuntimeError("Unexpected snapshot format or bucket scope.")
    if model["name"] != MODEL_NAME or model["version"] != 17 or model["run_id"] != RUN_ID:
        raise RuntimeError("Unexpected snapshot model identity.")
    pairs = set()
    for index, record in enumerate(manifest["objects"]):
        pair = (record["bucket"], record["key"])
        if record["bucket"] not in BUCKETS or pair in pairs or record["size"] < 0:
            raise RuntimeError("Invalid or duplicate snapshot object.")
        if record["entry"] != f"objects/{index:08d}":
            raise RuntimeError("Unexpected archive entry path.")
        pairs.add(pair)
    uri = urlsplit(model["download_uri"])
    mlmodel_key = uri.path.lstrip("/").rstrip("/") + "/MLmodel"
    if uri.scheme != "s3" or (uri.netloc, mlmodel_key) not in pairs:
        raise RuntimeError("Champion MLmodel metadata is absent from the snapshot.")


def inventory(s3, client):
    result = {
        "format_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "buckets": list(BUCKETS), "model": model_identity(client),
        "objects": object_inventory(s3),
    }
    validate_manifest(result)
    return result


class HashReader:
    def __init__(self, body):
        self.body = body
        self.digest = hashlib.sha256()

    def read(self, size):
        data = self.body.read(size)
        self.digest.update(data)
        return data


def export(s3, client, expected, output):
    validate_manifest(expected)
    if inventory(s3, client)["objects"] != expected["objects"] or model_identity(client) != expected["model"]:
        raise RuntimeError("Source changed after the inventory; retry during a quiet interval.")
    records = []
    with tarfile.open(fileobj=output, mode="w|") as archive:
        for index, record in enumerate(expected["objects"]):
            response = s3.get_object(Bucket=record["bucket"], Key=record["key"], IfMatch=record["etag"])
            with closing(response["Body"]) as body:
                if response["ContentLength"] != record["size"] or response["ETag"] != record["etag"]:
                    raise RuntimeError("Object changed during export.")
                reader = HashReader(body)
                entry = tarfile.TarInfo(record["entry"])
                entry.size = record["size"]
                entry.mode = 0o600
                archive.addfile(entry, reader)
                headers = {name: response[name] for name in (
                    "ContentType", "ContentEncoding", "ContentDisposition", "ContentLanguage",
                    "CacheControl", "Metadata",
                ) if name in response}
                records.append({**record, "sha256": reader.digest.hexdigest(), "headers": headers})
            if (index + 1) % 20 == 0 or index + 1 == len(expected["objects"]):
                print(f"S3: {index + 1}/{len(expected['objects'])} objetos copiados", file=sys.stderr, flush=True)
        current = inventory(s3, client)
        if current["objects"] != expected["objects"] or current["model"] != expected["model"]:
            raise RuntimeError("Source changed during export; snapshot was not finalized.")
        payload = json.dumps({**expected, "objects": records}).encode()
        footer = tarfile.TarInfo("manifest.json")
        footer.size = len(payload)
        footer.mode = 0o600
        archive.addfile(footer, io.BytesIO(payload))


def stream_hash(source):
    digest = hashlib.sha256()
    while True:
        data = source.read(4 * 1024 * 1024)
        if not data:
            return digest.hexdigest()
        digest.update(data)


def file_hash(path):
    with path.open("rb") as source:
        return stream_hash(source)


def verify(directory):
    dump = directory / "postgres.dump"
    with dump.open("rb") as source:
        if source.read(5) != b"PGDMP":
            raise RuntimeError("Expected a PostgreSQL custom-format dump.")
    digests = {}
    manifest = None
    with tarfile.open(directory / "objects.tar", mode="r|") as archive:
        for member in archive:
            if not member.isfile() or manifest is not None or member.name in digests:
                raise RuntimeError("Invalid snapshot archive structure.")
            with closing(archive.extractfile(member)) as data:
                if member.name == "manifest.json":
                    if member.size > 16 * 1024 * 1024:
                        raise RuntimeError("Snapshot manifest exceeds the supported size.")
                    manifest = json.load(data)
                elif member.name == f"objects/{len(digests):08d}":
                    digests[member.name] = (member.size, stream_hash(data))
                else:
                    raise RuntimeError("Unexpected snapshot archive path.")
    if manifest is None:
        raise RuntimeError("Snapshot is incomplete: no manifest footer.")
    validate_manifest(manifest)
    if len(digests) != len(manifest["objects"]):
        raise RuntimeError("Snapshot object count mismatch.")
    for record in manifest["objects"]:
        if digests[record["entry"]] != (record["size"], record["sha256"]):
            raise RuntimeError("Snapshot object hash/size mismatch.")
    summary = {
        "snapshot": directory.name, "verified_at": datetime.now(timezone.utc).isoformat(),
        "model": manifest["model"],
        "buckets": {bucket: {
            "objects": sum(item["bucket"] == bucket for item in manifest["objects"]),
            "bytes": sum(item["size"] for item in manifest["objects"] if item["bucket"] == bucket),
        } for bucket in BUCKETS},
        "files": {name: {"bytes": (directory / name).stat().st_size, "sha256": file_hash(directory / name)}
                  for name in ("postgres.dump", "objects.tar")},
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (directory / "SHA256SUMS").write_text("".join(
        f"{data['sha256']}  {name}\n" for name, data in summary["files"].items()
    ))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "export", "verify"))
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.mode == "verify":
        if args.directory is None:
            parser.error("verify requires --directory")
        print(json.dumps(verify(args.directory), indent=2))
        return
    import boto3
    from botocore.config import Config
    from mlflow import MlflowClient

    endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    if endpoint != "http://rustfs:9000":
        raise RuntimeError("Expected cluster-internal RustFS endpoint.")
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "15"
    client = MlflowClient(tracking_uri="http://127.0.0.1:5000", registry_uri="http://127.0.0.1:5000")
    s3 = boto3.client("s3", endpoint_url=endpoint, config=Config(
        signature_version="s3v4", s3={"addressing_style": "path"},
        connect_timeout=5, read_timeout=30, retries={"max_attempts": 2},
    ))
    if args.mode == "inventory":
        print(json.dumps(inventory(s3, client), indent=2))
    else:
        export(s3, client, json.load(sys.stdin), sys.stdout.buffer)


if __name__ == "__main__":
    main()
