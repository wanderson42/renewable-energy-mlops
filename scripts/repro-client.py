"""Foreground launcher and read-only client gate for the isolated rehearsal."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repro_serving", ROOT / "scripts/repro-serving.py")
SERVING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVING)
PORTS = {"api": 18000, "mlflow": 15000, "rustfs": 19000, "prefect": 14200, "dashboard": 18501}


def urls():
    return {name: f"http://127.0.0.1:{port}" for name, port in PORTS.items()}


def credentials(deployment):
    container = next(c for c in deployment["spec"]["template"]["spec"]["containers"] if c["name"] == "mlflow")
    values = {item["name"]: item.get("value") for item in container["env"]}
    result = (values.get("AWS_ACCESS_KEY_ID"), values.get("AWS_SECRET_ACCESS_KEY"))
    if not all(isinstance(value, str) and value for value in result):
        raise RuntimeError("Credenciais explícitas do RustFS não encontradas no MLflow do ensaio.")
    return result


def child_environment(user, password):
    # Settings are scoped to children; never modify .env or a global Prefect profile.
    env = {k: v for k, v in os.environ.items() if not k.startswith("PREFECT_")
           and k not in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN")}
    target = urls()
    home = ROOT / ".repro/client/prefect"
    database = "sqlite+aiosqlite:///" + str(home / "orchestration.db")
    env.update({"API_URL": target["api"], "API_RELOAD_URL": target["api"] + "/reload-model",
                "MLFLOW_TRACKING_URI": target["mlflow"], "MLFLOW_REGISTRY_URI": target["mlflow"],
                "MLFLOW_HTTP_REQUEST_TIMEOUT": "15", "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2",
                "RUSTFS_ENDPOINT": target["rustfs"], "RUSTFS_BUCKET": "energy-lake",
                "RUSTFS_ROOT_USER": user, "RUSTFS_ROOT_PASSWORD": password,
                "AWS_ACCESS_KEY_ID": user, "AWS_SECRET_ACCESS_KEY": password,
                "MLFLOW_S3_ENDPOINT_URL": target["rustfs"], "AWS_ENDPOINT_URL": target["rustfs"],
                "AWS_ENDPOINT_URL_S3": target["rustfs"], "AWS_DEFAULT_REGION": "us-east-1",
                "AWS_EC2_METADATA_DISABLED": "true",
                "PREFECT_HOME": str(home), "PREFECT_PROFILES_PATH": str(home / "profiles.toml"),
                "PREFECT_PROFILE": "repro", "PREFECT_API_URL": target["prefect"] + "/api",
                "PREFECT_UI_URL": target["prefect"], "PREFECT_SERVER_UI_API_URL": target["prefect"] + "/api",
                "PREFECT_SERVER_DATABASE_CONNECTION_URL": database, "PREFECT_API_DATABASE_CONNECTION_URL": database,
                # Empty auth strings are NOT disabled auth in Prefect 3.8.6.
                # Leave these keys absent; the effective-settings gate checks None.
                "PREFECT_SERVER_API_HOST": "127.0.0.1", "PREFECT_SERVER_API_PORT": str(PORTS["prefect"]),
                "PREFECT_SERVER_UI_ENABLED": "true", "PREFECT_SERVER_EPHEMERAL_ENABLED": "false",
                "PREFECT_SERVER_ANALYTICS_ENABLED": "false", "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"})
    return env


def commands():
    result = {}
    for name, service, remote in (("api", "energy-api", 8000), ("mlflow", "mlflow", 5000), ("rustfs", "rustfs", 9000)):
        result[name] = SERVING.kubectl("port-forward", "--address", "127.0.0.1", "svc/" + service, f"{PORTS[name]}:{remote}")
    result["prefect"] = ["poetry", "run", "prefect", "server", "start", "--host", "127.0.0.1", "--port", str(PORTS["prefect"])]
    result["dashboard"] = ["poetry", "run", "streamlit", "run", "src/energy_mlops/app/app.py",
                           "--server.address=127.0.0.1", f"--server.port={PORTS['dashboard']}", "--server.headless=true"]
    return result


def require_free_ports():
    sockets = []
    try:
        for name, port in PORTS.items():
            listener = socket.socket()
            sockets.append(listener)
            try:
                listener.bind(("127.0.0.1", port))
            except OSError:
                raise RuntimeError(f"Porta {port} ({name}) ocupada; nenhum processo existente será encerrado.") from None
    finally:
        for listener in sockets:
            listener.close()


def fetch(url, json_response=False, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    with build_opener(ProxyHandler({})).open(Request(url, data=data, headers=headers), timeout=3) as response:
        if response.status != 200:
            raise RuntimeError("HTTP não retornou 200.")
        return json.load(response) if json_response else response.read()


def check_prefect_settings(prefect):
    target = urls()["prefect"]
    expected_home = ROOT / ".repro/client/prefect"
    expected_db = "sqlite+aiosqlite:///" + str(expected_home / "orchestration.db")
    database = prefect.server.database.connection_url
    if (Path(prefect.home).resolve() != expected_home or prefect.api.url != target + "/api"
            or prefect.server.ui.api_url != target + "/api"
            or database is None or database.get_secret_value() != expected_db):
        raise RuntimeError("Configuração efetiva do Prefect não está isolada.")
    # The server tests `is not None`, whereas /health bypasses authentication.
    if any(value is not None for value in (prefect.api.key, prefect.api.auth_string,
                                           prefect.server.api.auth_string)):
        raise RuntimeError("Autenticação Prefect definida no ensaio; verifique a configuração local sem publicar segredos.")


def check_prefect_ui():
    target = urls()["prefect"]
    settings = fetch(target + "/ui-settings", True)
    if settings.get("api_url") != target + "/api" or settings.get("auth") is not None:
        raise RuntimeError("Configuração HTTP da UI Prefect diverge do ensaio sem autenticação.")
    # This POST is a read-only count query, not a flow run creation.
    count = fetch(target + "/api/flow_runs/count", True, payload={})
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise RuntimeError("Consulta de dados Prefect retornou uma contagem inválida.")
    return {"api_url": settings["api_url"], "auth": None,
            "flow_runs_count": count, "data_query_http_status": 200,
            "browser_e2e": "not exercised by this gate"}


def require_children_alive(children):
    for name, process in children:
        if process.poll() is not None:
            raise RuntimeError(f"Processo {name} terminou. Consulte o log privado em .repro/client/{name}.log.")


def stop_children(children):
    # Each group was created by this invocation. There is no global pkill/PID file.
    for _, process in reversed(children):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _, process in reversed(children):
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=8)


def wait_services(children):
    target = urls()
    endpoints = [target["api"] + "/health", target["mlflow"] + "/health",
                 target["prefect"] + "/api/health", target["dashboard"] + "/_stcore/health"]
    deadline = time.monotonic() + 180
    while True:
        require_children_alive(children)
        try:
            for endpoint in endpoints:
                fetch(endpoint)
            return
        except (URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise RuntimeError("Serviços não ficaram saudáveis em 180s; consulte os logs privados.") from None
            time.sleep(0.5)


def environment_from_target():
    SERVING.check_target()
    deployment = json.loads(SERVING.run(SERVING.kubectl("get", "deployment", "mlflow", "-o", "json"), capture=True))
    return child_environment(*credentials(deployment))


def run_probe(env):
    result = subprocess.run(["poetry", "run", "python", str(Path(__file__).resolve()), "probe"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
    log = ROOT / ".repro/client/probe.log"
    log.write_text(result.stdout + "\n" + result.stderr)
    log.chmod(0o600)
    if result.returncode != 0:
        raise RuntimeError("Gate do cliente falhou. Inspecione .repro/client/probe.log; não publique sua saída bruta.")
    receipt = json.loads(result.stdout)
    SERVING.write_private(ROOT / ".repro/client-validation.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def start():
    for cli in ("poetry", "kubectl"):
        if not shutil.which(cli):
            raise RuntimeError(f"{cli} ausente.")
    env = environment_from_target()
    require_free_ports()
    home = Path(env["PREFECT_HOME"])
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    home.chmod(0o700)
    home.parent.chmod(0o700)
    profile = Path(env["PREFECT_PROFILES_PATH"])
    if not profile.exists():
        profile.write_text('active = "repro"\n\n[profiles.repro]\n')
        profile.chmod(0o600)
    children, logs = [], []
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        for name, command in commands().items():
            path = home.parent / (name + ".log")
            log = path.open("w")
            path.chmod(0o600)
            logs.append(log)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            children.append((name, process))
        print("Inicializando o cliente de ensaio; logs privados em .repro/client/", flush=True)
        wait_services(children)
        run_probe(env)
        print("Cliente pronto. Dashboard: " + urls()["dashboard"] + "; Prefect: " + urls()["prefect"], flush=True)
        print("Mantenha este terminal aberto. Ctrl+C encerra os processos desta execução.", flush=True)
        while True:
            require_children_alive(children)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Encerrando o cliente do ensaio...", flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        try:
            stop_children(children)
        finally:
            for log in logs:
                log.close()


def probe():
    # Run inside Poetry, using the exact app environment, not the MLflow image.
    import tomllib
    from importlib.metadata import version
    import boto3
    from botocore.config import Config
    from mlflow import MlflowClient
    from prefect.settings import get_current_settings
    from energy_mlops.config import settings

    target = urls()
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError("O cliente precisa do Python 3.14 do ambiente Poetry.")
    locked = {p["name"]: p["version"] for p in tomllib.loads((ROOT / "poetry.lock").read_text())["package"]}
    runtime = {name: version(name) for name in ("prefect", "streamlit", "mlflow", "boto3", "botocore", "s3fs")}
    if any(value != locked[name] for name, value in runtime.items()):
        raise RuntimeError("Dependências do cliente divergem do poetry.lock; execute poetry install.")
    if (os.environ.get("API_URL"), settings.MLFLOW_TRACKING_URI, settings.RUSTFS_ENDPOINT, settings.PREFECT_API_URL) != (
        target["api"], target["mlflow"], target["rustfs"], target["prefect"] + "/api"
    ):
        raise RuntimeError("Endpoints efetivos da aplicação fora do ensaio.")
    prefect = get_current_settings()
    expected_home = ROOT / ".repro/client/prefect"
    check_prefect_settings(prefect)
    for url in (target["mlflow"] + "/health", target["prefect"] + "/api/health", target["dashboard"] + "/_stcore/health"):
        fetch(url)
    if not (expected_home / "orchestration.db").is_file():
        raise RuntimeError("Banco SQLite do Prefect não encontrado no diretório do ensaio.")
    prefect_ui = check_prefect_ui()
    health = fetch(target["api"] + "/health", True)
    if health.get("status") != "healthy" or health.get("model_loaded") is not True:
        raise RuntimeError("API do ensaio sem modelo saudável.")
    info = fetch(target["api"] + "/model-info", True)
    SERVING.check_identity(info)
    client = MlflowClient(tracking_uri=target["mlflow"], registry_uri=target["mlflow"])
    model = client.get_model_version_by_alias(SERVING.MODEL, "champion")
    if int(model.version) != 17 or model.run_id != SERVING.RUN or client.get_run(model.run_id).info.run_id != SERVING.RUN:
        raise RuntimeError("Registry/Run do cliente divergem da v17 servida.")
    manifest = json.loads((ROOT / ".repro/restore-manifest.json").read_text())
    if client.get_model_version_download_uri(model.name, model.version) != manifest["model"]["download_uri"]:
        raise RuntimeError("URI do artefato diverge do restore.")
    key = manifest["model"]["download_uri"].split("s3://mlflow-artifacts/", 1)[1].rstrip("/") + "/MLmodel"
    expected = next(r for r in manifest["objects"] if r["bucket"] == "mlflow-artifacts" and r["key"] == key)
    s3 = boto3.client("s3", endpoint_url=target["rustfs"], aws_access_key_id=settings.RUSTFS_ROOT_USER,
                      aws_secret_access_key=settings.RUSTFS_ROOT_PASSWORD, config=Config(
                          signature_version="s3v4", s3={"addressing_style": "path"}, connect_timeout=5, read_timeout=15, retries={"max_attempts": 2}))
    for bucket in ("energy-lake", "mlflow-artifacts"):
        s3.head_bucket(Bucket=bucket)
    body = s3.get_object(Bucket="mlflow-artifacts", Key=key)["Body"]
    try:
        data = body.read(1048577)
    finally:
        body.close()
    if len(data) > 1048576 or len(data) != expected["size"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise RuntimeError("Artefato lido pelo cliente diverge do manifesto restaurado.")
    if fetch(target["api"] + "/model-info", True) != info:
        raise RuntimeError("Identidade da API mudou durante o gate do cliente.")
    print(json.dumps({"validated_at": datetime.now(timezone.utc).isoformat(), "urls": target,
                      "context": "kind-energy-mlops-repro", "namespace": "energy-mlops-repro",
                      "python": sys.version.split()[0], "packages": runtime, "lock_versions_match": True,
                      "model": info, "registry_run_verified": True, "buckets_accessible": ["energy-lake", "mlflow-artifacts"],
                      "artifact": {"key": key, "bytes": len(data), "sha256": expected["sha256"]},
                      "prefect_database": ".repro/client/prefect/orchestration.db", "prefect_settings_isolated": True,
                      "prefect_ui": prefect_ui,
                      "dashboard_health_http_status": 200, "dashboard_browser_e2e": "not exercised by this gate",
                      "scope": "local services, effective app settings and read-only client integrations"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("start", "check", "probe"), default="start", nargs="?")
    args = parser.parse_args()
    if args.mode == "probe":
        probe()
    elif args.mode == "check":
        env = environment_from_target()
        if not Path(env["PREFECT_PROFILES_PATH"]).exists():
            raise RuntimeError("Inicie make repro-client em outro terminal primeiro.")
        run_probe(env)
    else:
        start()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        message = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        print("Cliente de ensaio falhou: " + message, file=sys.stderr)
        sys.exit(1)
