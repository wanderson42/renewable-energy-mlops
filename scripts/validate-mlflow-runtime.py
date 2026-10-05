"""Read-only gate for the packaged MLflow server in the isolated rehearsal."""

import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repro_serving", ROOT / "scripts/repro-serving.py")
SERVING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVING)
BUILD = json.loads((ROOT / "docs/evidence/mlflow_runtime_build_2026-10-05.json").read_text())
IMAGE = BUILD["published_image"]["reference"]


def check_deployment(deployment, pods):
    containers = deployment["spec"]["template"]["spec"]["containers"]
    container = next(item for item in containers if item["name"] == "mlflow")
    if container["image"] != IMAGE or "pip install" in " ".join(container.get("command", [])):
        raise RuntimeError("Deployment MLflow não usa o runtime empacotado esperado.")
    digest = IMAGE.split("@", 1)[1]
    for pod in pods["items"]:
        for status in pod.get("status", {}).get("containerStatuses", []):
            if status.get("name") == "mlflow" and status.get("ready") and status.get("imageID", "").endswith("@" + digest):
                if any(c.get("name") == "mlflow" and c.get("image") == IMAGE for c in pod["spec"]["containers"]):
                    return status["imageID"]
    raise RuntimeError("Pod Ready do MLflow com o digest esperado não encontrado.")


def expected_artifact(manifest):
    uri = urlsplit(manifest["model"]["download_uri"])
    key = uri.path.lstrip("/").rstrip("/") + "/MLmodel"
    if uri.scheme != "s3" or uri.netloc != "mlflow-artifacts":
        raise RuntimeError("URI do modelo fora do artifact store esperado.")
    records = [item for item in manifest["objects"] if item["bucket"] == uri.netloc and item["key"] == key]
    if len(records) != 1 or not 0 < records[0]["size"] <= 1048576 or not re.fullmatch(r"[a-f0-9]{64}", records[0]["sha256"]):
        raise RuntimeError("Metadados MLmodel ausentes ou inválidos no manifesto restaurado.")
    return {"bucket": uri.netloc, "key": key, "bytes": records[0]["size"], "sha256": records[0]["sha256"]}


def check_runtime(runtime):
    if any(runtime.get(field) != BUILD["runtime"][field] for field in ("python", "packages", "imports_passed", "pip_check_passed")):
        raise RuntimeError("Runtime do pod diverge da imagem aprovada no build.")


def main():
    SERVING.check_target()
    SERVING.run(SERVING.kubectl("rollout", "status", "deployment/mlflow", "--timeout=300s"), timeout=330)
    deployment = json.loads(SERVING.run(SERVING.kubectl("get", "deployment", "mlflow", "-o", "json"), capture=True))
    pods = json.loads(SERVING.run(SERVING.kubectl("get", "pods", "-l", "app=mlflow", "-o", "json"), capture=True))
    image_id = check_deployment(deployment, pods)
    SERVING.check_restore()  # Live Registry query through the server backed by PostgreSQL.
    checkpoint = SERVING.load_checkpoint()  # Checks the two current PVC UIDs against the saved checkpoint.
    runtime = json.loads(SERVING.run(SERVING.kubectl(
        "exec", "deployment/mlflow", "--", "python3", "/opt/energy-mlops-mlflow/verify_runtime.py"
    ), capture=True, timeout=90))
    check_runtime(runtime)
    manifest = json.loads((ROOT / ".repro/restore-manifest.json").read_text())
    restored = json.loads((ROOT / ".repro/restored-data.json").read_text())
    if manifest["model"] != restored["model"]:
        raise RuntimeError("Manifesto e comprovante de restore divergem.")
    expected = expected_artifact(manifest)
    # Fetch only the small MLmodel metadata file, never unpickle the model in
    # the server's Python environment. Credentials remain inside the pod.
    code = '''import hashlib, json, os, sys
import boto3
from botocore.config import Config
expected = json.loads(sys.argv[1])
if os.environ.get("MLFLOW_S3_ENDPOINT_URL") != "http://rustfs:9000":
    raise RuntimeError("Unexpected RustFS endpoint")
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
s3 = boto3.client("s3", endpoint_url="http://rustfs:9000", config=Config(
    signature_version="s3v4", s3={"addressing_style": "path"},
    connect_timeout=5, read_timeout=15, retries={"max_attempts": 2}))
response = s3.get_object(Bucket=expected["bucket"], Key=expected["key"])
try:
    data = response["Body"].read(1048577)
finally:
    response["Body"].close()
if len(data) > 1048576:
    raise RuntimeError("MLmodel exceeds the size limit")
print(json.dumps({"bucket": expected["bucket"], "key": expected["key"],
    "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}))
'''
    artifact = json.loads(SERVING.run(SERVING.kubectl(
        "exec", "deployment/mlflow", "--", "python3", "-c", code, json.dumps(expected)
    ), capture=True, timeout=90))
    if artifact != expected:
        raise RuntimeError("Bytes/hash do artefato divergem do manifesto restaurado.")
    SERVING.check_restore()
    result = {"validated_at": datetime.now(timezone.utc).isoformat(),
              "context": "kind-energy-mlops-repro", "namespace": "energy-mlops-repro",
              "image": IMAGE, "pod_image_id": image_id, "runtime": runtime,
              "registry_model": restored["model"], "artifact": artifact,
              "pvc_identity_preserved": SERVING.volume_identity() == checkpoint["pvc_identity"],
              "runtime_installation_at_startup": False,
              "scope": "MLflow runtime, live Registry query and one MLmodel artifact read"}
    if not result["pvc_identity_preserved"]:
        raise RuntimeError("PVCs mudaram durante o gate.")
    SERVING.write_private(ROOT / ".repro/mlflow-runtime-validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        message = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        print(f"Gate MLflow falhou: {message}", file=sys.stderr)
        sys.exit(1)
