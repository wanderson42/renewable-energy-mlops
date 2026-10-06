"""Enable the restored rehearsal through Terraform and compare serving over HTTP."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
MODEL = "ensemble_lgb_xgb_rf_bahia"
RUN = "1d13a61244c54f06aa70f43a9993ea37"
IMAGE = "ghcr.io/wanderson42/renewable-energy-mlops@sha256:95208ab282e24014a81f60d1e3eac01b8db36e40f05ae25e9f168264e4b7c188"
DEFAULT_DIGEST = IMAGE.split("@", 1)[1]
FAILURE_DIGEST = "sha256:" + "0" * 64
SOURCE = "http://localhost:8000"
PAYLOAD = {
    "predictions": [
        {"date": "2026-10-04T10:00:00Z", "wind_speed_100m": 9.0,
         "wind_direction_100m": 150.0, "temperature_2m": 25.0},
        {"date": "2026-10-04T11:00:00Z", "wind_speed_100m": 10.0,
         "wind_direction_100m": 170.0, "temperature_2m": 26.0},
        {"date": "2026-10-04T12:00:00Z", "wind_speed_100m": 12.5,
         "wind_direction_100m": 180.0, "temperature_2m": 27.0},
    ]
}


def run(args, capture=False, **kwargs):
    result = subprocess.run(args, check=True, text=True, capture_output=capture,
                            timeout=kwargs.pop("timeout", 60), **kwargs)
    return result.stdout if capture else None


def kubectl(*args):
    return ["kubectl", "--kubeconfig", str(ROOT / ".repro/kubeconfig"),
            "--context", "kind-energy-mlops-repro", "--namespace", "energy-mlops-repro", *args]


def terraform(*args):
    return [str(ROOT / ".repro/bin/terraform"),
            f"-chdir={ROOT / 'terraform/environments/local'}", *args]


def configuration(value=None):
    if value is None:
        path = ROOT / ".repro/deployment.tfvars.json"
        value = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(value, dict):
        raise RuntimeError("Configuração do deployment inválida.")
    result = {"api_enabled": value.get("api_enabled", True),
              "api_digest": value.get("api_digest", DEFAULT_DIGEST),
              "deployment_timeout_seconds": value.get("deployment_timeout_seconds", 600)}
    timeout = result["deployment_timeout_seconds"]
    if result["api_enabled"] is not True or not isinstance(result["api_digest"], str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", result["api_digest"]):
        raise RuntimeError("Configuração de serving precisa de API habilitada e digest completo.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout != int(timeout) or not 60 <= timeout <= 900:
        raise RuntimeError("Timeout de serving inválido.")
    return result


def write_private(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def check_target():
    nodes = run(kubectl("get", "nodes", "-o", "jsonpath={.items[*].metadata.name}"), capture=True)
    if nodes.strip() != "energy-mlops-repro-control-plane":
        raise RuntimeError("Node do ensaio não confirmado.")


def check_restore():
    receipt = json.loads((ROOT / ".repro/restored-data.json").read_text())
    model = receipt["model"]
    if (model["name"], model["version"], model["run_id"]) != (MODEL, 17, RUN):
        raise RuntimeError("Comprovante de restauração não corresponde à v17.")
    if receipt["verified_objects"] != 251 or receipt["verified_bytes"] != 1829741640:
        raise RuntimeError("Comprovante de restauração não corresponde ao snapshot validado.")
    # Read the current destination Registry too; a stale local receipt is insufficient.
    code = '''import json, os
from mlflow import MlflowClient
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "15"
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "2"
c = MlflowClient(tracking_uri="http://127.0.0.1:5000", registry_uri="http://127.0.0.1:5000")
m = c.get_model_version_by_alias("ensemble_lgb_xgb_rf_bahia", "champion")
print(json.dumps({"name": m.name, "version": int(m.version), "run_id": m.run_id, "source": m.source,
 "download_uri": c.get_model_version_download_uri(m.name, m.version)}))
'''
    live = json.loads(run(kubectl("exec", "deployment/mlflow", "--", "python3", "-c", code), capture=True))
    if live != model:
        raise RuntimeError("Registry do ensaio diverge da identidade restaurada.")


def inspect_plan(plan):
    """Reject replacement, a bootstrap install or a plan for any other resource."""
    # CLI inputs can retain the string "true"; validate Terraform's evaluated output.
    target = plan.get("planned_values", {}).get("outputs", {}).get("deployment_target", {}).get("value", {})
    if plan.get("errored") or target.get("api_enabled") is not True:
        raise RuntimeError("O plano precisa habilitar explicitamente a API.")
    if (target.get("context"), target.get("namespace"), target.get("release")) != (
        "kind-energy-mlops-repro", "energy-mlops-repro", "energy-mlops-repro",
    ):
        raise RuntimeError("O output planejado não corresponde ao ensaio.")
    changes = plan.get("resource_changes", [])
    if len(changes) != 1:
        raise RuntimeError("Esperado apenas o recurso helm_release.mlops.")
    item = changes[0]
    change = item["change"]
    if item["address"] != "helm_release.mlops" or change["actions"] not in (["update"], ["no-op"]):
        raise RuntimeError("Somente atualização da release existente é permitida.")
    for side in ("before", "after"):
        resource = change[side]
        if resource["name"] != "energy-mlops-repro" or resource["namespace"] != "energy-mlops-repro":
            raise RuntimeError("Release/namespace fora do ensaio.")
    settings = {item["name"]: item["value"] for item in change["after"]["set"]}
    if settings.get("api.enabled") != "true":
        raise RuntimeError("A configuração Helm planejada não habilita a API.")
    config = configuration(target)
    if settings.get("api.image.digest") != config["api_digest"] or change["after"]["timeout"] != config["deployment_timeout_seconds"]:
        raise RuntimeError("Digest/timeout do plano divergem do output avaliado.")
    return config


def serving_plan(digest=None, timeout=600):
    check_target()
    check_restore()
    command = ["bash", str(ROOT / "scripts/plan-repro.sh"), "--serving"]
    if digest is not None:
        configuration({"api_digest": digest, "deployment_timeout_seconds": timeout})
        command.extend([digest, str(timeout)])
    run(command, timeout=600)
    plan = json.loads(run(terraform("show", "-json", str(ROOT / ".repro/serving.tfplan")), capture=True))
    inspect_plan(plan)
    print("Plano de serving verificado. Nenhum workload foi aplicado.")
    return plan


def serving_apply():
    check_target()
    check_restore()
    plan = json.loads(run(terraform("show", "-json", str(ROOT / ".repro/serving.tfplan")), capture=True))
    config = inspect_plan(plan)
    # Persist the authorized intent even if apply/gate fails, avoiding accidental removal later.
    write_private(ROOT / ".repro/deployment.tfvars.json", config)
    print("Aplicando no ensaio. Log bruto privado: .repro/terraform-apply.log", flush=True)
    apply_log = ROOT / ".repro/terraform-apply.log"
    apply_log.touch(mode=0o600, exist_ok=True)
    apply_log.chmod(0o600)
    with apply_log.open("w") as log:
        run(terraform("apply", "-input=false", str(ROOT / ".repro/serving.tfplan")),
            stdout=log, stderr=subprocess.STDOUT, timeout=900)
    print("Terraform apply concluído. Iniciando o gate de serving.", flush=True)
    return validate_serving()


def request(base, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(base + path, data=data, headers={"Content-Type": "application/json"})
    # Both endpoints are local; do not route requests through an environment proxy.
    with build_opener(ProxyHandler({})).open(req, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("Endpoint não retornou HTTP 200.")
        return json.load(response)


def check_identity(info):
    if (info.get("model_name"), info.get("alias"), info.get("version"), info.get("run_id")) != (MODEL, "champion", 17, RUN):
        raise RuntimeError("API não serve a identidade champion v17 esperada.")


def compare_predictions(source, destination):
    if not isinstance(source, list) or not isinstance(destination, list) or len(source) != 3 or len(destination) != 3:
        raise RuntimeError("Quantidade de predições diverge do lote fixo.")
    differences = {"predicted_fc": 0.0, "predicted_mw": 0.0}
    for expected, left, right in zip(PAYLOAD["predictions"], source, destination):
        for prediction in (left, right):
            timestamp = datetime.fromisoformat(prediction["date"].replace("Z", "+00:00"))
            if timestamp != datetime.fromisoformat(expected["date"].replace("Z", "+00:00")):
                raise RuntimeError("Datas/ordem das predições divergem do payload.")
            for field in differences:
                value = prediction[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise RuntimeError("Predição ausente, negativa ou não finita.")
            if prediction["predicted_fc"] > 1:
                raise RuntimeError("Fator de capacidade fora de [0, 1].")
        for field, tolerance in (("predicted_fc", 1e-12), ("predicted_mw", 1e-9)):
            difference = abs(left[field] - right[field])
            differences[field] = max(differences[field], difference)
            if difference > tolerance:
                raise RuntimeError(f"Inferência diverge entre origem e ensaio: {field}.")
    return differences


def compare_endpoints(destination):
    for base in (SOURCE, destination):
        health = request(base, "/health")
        if health.get("status") != "healthy" or health.get("model_loaded") is not True:
            raise RuntimeError("API sem modelo saudável.")
    left = request(SOURCE, "/model-info")
    right = request(destination, "/model-info")
    check_identity(left)
    check_identity(right)
    if not isinstance(left.get("metrics"), dict) or not left["metrics"] or left["metrics"] != right.get("metrics"):
        raise RuntimeError("Métricas servidas divergem entre origem e destino.")
    source = request(SOURCE, "/predict/batch", PAYLOAD)
    target = request(destination, "/predict/batch", PAYLOAD)
    differences = compare_predictions(source, target)
    if request(SOURCE, "/model-info") != left or request(destination, "/model-info") != right:
        raise RuntimeError("Identidade/metadados mudaram durante a comparação.")
    return {"validated_at": datetime.now(timezone.utc).isoformat(),
            "model": right, "payload": PAYLOAD, "source_predictions": source,
            "destination_predictions": target, "max_absolute_difference": differences,
            "absolute_tolerances": {"predicted_fc": 1e-12, "predicted_mw": 1e-9},
            "reference": "paired_synthetic_batch_v1"}


def validate_serving():
    check_target()
    expected_image = IMAGE.split("@", 1)[0] + "@" + configuration()["api_digest"]
    run(kubectl("rollout", "status", "deployment/energy-api", "--timeout=300s"), timeout=330)
    deployment = json.loads(run(kubectl("get", "deployment", "energy-api", "-o", "json"), capture=True))
    if [c["image"] for c in deployment["spec"]["template"]["spec"]["containers"]] != [expected_image]:
        raise RuntimeError("Deployment não usa o digest de referência.")
    pods = json.loads(run(kubectl("get", "pods", "-l", "app=energy-api", "-o", "json"), capture=True))
    active = [p for p in pods["items"] if not p["metadata"].get("deletionTimestamp")]
    if len(active) != 1:
        raise RuntimeError("Esperado um pod ativo da API no ensaio.")
    statuses = active[0]["status"].get("containerStatuses", [])
    if len(statuses) != 1 or not statuses[0]["ready"] or not statuses[0]["imageID"].endswith(expected_image.split("@", 1)[1]):
        raise RuntimeError("Pod não confirmou readiness e digest esperado.")
    log_path = ROOT / ".repro/serving-port-forward.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(kubectl("port-forward", "--address=127.0.0.1", "service/energy-api", ":8000"),
                                   stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.monotonic() + 30
            destination = None
            while time.monotonic() < deadline:
                match = re.search(r"Forwarding from 127\.0\.0\.1:(\d+) -> 8000", log_path.read_text())
                if match:
                    destination = f"http://127.0.0.1:{match.group(1)}"
                    break
                if process.poll() is not None:
                    raise RuntimeError("Port-forward terminou; consulte .repro/serving-port-forward.log.")
                time.sleep(0.2)
            if destination is None:
                raise RuntimeError("Port-forward não iniciou em 30s.")
            result = compare_endpoints(destination)
            result["image"] = expected_image
            result["context"] = "kind-energy-mlops-repro"
            result["namespace"] = "energy-mlops-repro"
            (ROOT / ".repro/serving-validation.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            return result
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def volume_identity():
    volumes = json.loads(run(kubectl("get", "pvc", "postgres-pvc", "rustfs-pvc", "-o", "json"), capture=True))
    result = {}
    for item in volumes["items"]:
        if item["status"]["phase"] != "Bound":
            raise RuntimeError("PVC do ensaio não está Bound.")
        result[item["metadata"]["name"]] = item["metadata"]["uid"]
    if set(result) != {"postgres-pvc", "rustfs-pvc"}:
        raise RuntimeError("PVCs esperados não foram encontrados.")
    return result


def checkpoint():
    result = validate_serving()
    saved = {"configuration": configuration(), "model": result["model"],
             "image": result["image"], "validated_at": result["validated_at"],
             "pvc_identity": volume_identity()}
    write_private(ROOT / ".repro/rollback-checkpoint.json", saved)
    print("Checkpoint de rollback salvo após gate real de serving.")
    return saved


def load_checkpoint():
    saved = json.loads((ROOT / ".repro/rollback-checkpoint.json").read_text())
    config = configuration(saved["configuration"])
    check_identity(saved["model"])
    if saved["image"] != IMAGE.split("@", 1)[0] + "@" + config["api_digest"] or volume_identity() != saved["pvc_identity"]:
        raise RuntimeError("Checkpoint diverge da imagem ou dos PVCs do ensaio.")
    return saved


def require_image_only_change(plan):
    inspect_plan(plan)
    change = plan["resource_changes"][0]["change"]
    before, after = change["before"], change["after"]
    old_settings = {item["name"]: item["value"] for item in before["set"]}
    new_settings = {item["name"]: item["value"] for item in after["set"]}
    if any(before.get(key) != after.get(key) for key in ("chart", "version", "values")) or old_settings.get("api.enabled") != "true" or old_settings.get("repro.chartHash") != new_settings.get("repro.chartHash"):
        raise RuntimeError("O ensaio de recuperação permite somente digest/timeout, sem mudanças de chart ou inputs.")
    for name in set(old_settings) | set(new_settings):
        if name != "api.image.digest" and old_settings.get(name) != new_settings.get(name):
            raise RuntimeError("Setting fora do escopo do ensaio de recuperação.")


def failure_plan():
    check_target()
    load_checkpoint()
    plan = serving_plan(FAILURE_DIGEST, 60)
    require_image_only_change(plan)
    print("Plano de falha controlada: digest reservado indisponível, timeout Helm de 60s; somente no ensaio.")


def rollback():
    check_target()
    saved = load_checkpoint()
    write_private(ROOT / ".repro/deployment.tfvars.json", configuration(saved["configuration"]))
    plan = serving_plan()
    require_image_only_change(plan)
    result = serving_apply()
    if volume_identity() != saved["pvc_identity"]:
        raise RuntimeError("Identidade dos PVCs mudou durante a recuperação.")
    print("Rollback declarativo concluído com gate de serving e PVCs preservados.")
    return result


def recovery_test():
    checkpoint()
    failure_plan()
    observed = {"expected_pull_failure": False, "waiting_reasons": []}
    try:
        try:
            serving_apply()
        except subprocess.CalledProcessError:
            pods = json.loads(run(kubectl("get", "pods", "-l", "app=energy-api", "-o", "json"), capture=True))
            reasons = sorted({status.get("state", {}).get("waiting", {}).get("reason", "")
                              for pod in pods["items"] for status in pod.get("status", {}).get("containerStatuses", [])} - {""})
            observed = {"expected_pull_failure": bool(set(reasons) & {"ErrImagePull", "ImagePullBackOff"}), "waiting_reasons": reasons}
            write_private(ROOT / ".repro/recovery-failure.json", observed)
            failed_log = ROOT / ".repro/terraform-apply.log"
            if failed_log.exists():
                shutil.copyfile(failed_log, ROOT / ".repro/recovery-failed-apply.log")
                (ROOT / ".repro/recovery-failed-apply.log").chmod(0o600)
            print(json.dumps(observed, indent=2), flush=True)
    finally:
        # Always restore the declared checkpoint after attempting the controlled update.
        print("Restaurando o checkpoint pelo Terraform...", flush=True)
        restored = rollback()
    if not observed["expected_pull_failure"]:
        raise RuntimeError("Checkpoint recuperado, mas a falha de pull esperada não foi confirmada.")
    run(["bash", str(ROOT / "scripts/plan-repro.sh")], timeout=600)
    plan = json.loads(run(terraform("show", "-json", str(ROOT / ".repro/bootstrap.tfplan")), capture=True))
    inspect_plan(plan)
    if plan["resource_changes"][0]["change"]["actions"] != ["no-op"] or any(item["actions"] != ["no-op"] for item in plan.get("output_changes", {}).values()):
        raise RuntimeError("Plano posterior à recuperação ainda apresenta mudanças.")
    result = {"verified_at": datetime.now(timezone.utc).isoformat(), **observed,
              "recovered_image": restored["image"], "model": restored["model"],
              "max_absolute_difference": restored["max_absolute_difference"],
              "post_recovery_plan": "No changes", "pvc_identity_preserved": True,
              "scope": "controlled image-pull failure in the isolated rehearsal"}
    write_private(ROOT / ".repro/recovery-validation.json", result)
    print(json.dumps(result, indent=2))


def release_plan():
    digest = os.environ.get("REPRO_API_DIGEST", "")
    serving_plan(digest)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "apply", "validate", "checkpoint", "failure-plan", "rollback", "recovery-test", "release-plan"))
    args = parser.parse_args()
    try:
        {"plan": serving_plan, "apply": serving_apply, "validate": validate_serving,
         "checkpoint": checkpoint, "failure-plan": failure_plan, "rollback": rollback,
         "recovery-test": recovery_test, "release-plan": release_plan}[args.mode]()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"Serving interrompido: {exc}. Inspecione o ensaio; se a recuperação não terminou, use make repro-rollback.") from exc


if __name__ == "__main__":
    main()
