"""Validate the server image; --server-url also checks a temporary HTTP server."""

import argparse
import importlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from importlib.metadata import version
from pathlib import Path


def validate_packages():
    expected = dict(
        line.split("==", 1)
        for line in Path(__file__).with_name("requirements.txt").read_text().splitlines()
        if line and not line.startswith("#")
    )
    actual = {name: version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Package versions differ from the observed runtime: {actual}")
    if sys.version.split()[0] != "3.11.15":
        raise RuntimeError("Expected Python 3.11.15 from the pinned MLflow base image.")
    for module in ("mlflow", "boto3", "botocore", "psycopg2", "s3transfer",
                   "jmespath", "urllib3", "dateutil", "six"):
        importlib.import_module(module)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True,
                   capture_output=True, text=True, timeout=30)
    return {"python": sys.version.split()[0], "packages": actual,
            "imports_passed": True, "pip_check_passed": True}


def validate_http(server_url):
    # This check is exclusively for the network-isolated smoke container.
    if server_url != "http://127.0.0.1:5000":
        raise ValueError("The smoke check only accepts its container-local server.")
    deadline = time.monotonic() + 90
    while True:
        try:
            with urllib.request.urlopen(server_url + "/health", timeout=3) as response:
                if response.status != 200:
                    raise RuntimeError("MLflow health did not return HTTP 200.")
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise RuntimeError("MLflow did not become healthy within 90 seconds.") from None
            time.sleep(1)
    request = urllib.request.Request(
        server_url + "/api/2.0/mlflow/experiments/search",
        data=json.dumps({"max_results": 1}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        result = json.load(response)
    if not isinstance(result, dict) or not isinstance(result.get("experiments", []), list):
        raise RuntimeError("Unexpected experiment-search response.")
    return {"health_http_status": 200, "tracking_search_passed": True,
            "backend": "temporary SQLite", "external_network": "disabled"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url")
    args = parser.parse_args()
    result = validate_packages()
    if args.server_url:
        result["http_smoke"] = validate_http(args.server_url)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
