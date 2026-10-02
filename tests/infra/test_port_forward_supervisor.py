from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR = PROJECT_ROOT / "scripts" / "port-forward-supervisor.sh"


def _wait_until(
    predicate,
    *,
    timeout: float = 6.0,
    interval: float = 0.05,
) -> None:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if predicate():
            return

        time.sleep(interval)

    raise AssertionError(
        "Condição não satisfeita dentro do timeout."
    )


def _write_fake_kubectl(
    tmp_path: Path,
    body: str,
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    kubectl = bin_dir / "kubectl"
    kubectl.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        f"{body}\n",
        encoding="utf-8",
    )
    kubectl.chmod(0o755)

    return bin_dir


def _supervisor_env(
    bin_dir: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = (
        f"{bin_dir}{os.pathsep}{env['PATH']}"
    )
    return env


def test_supervisor_requires_three_arguments() -> None:
    result = subprocess.run(
        ["bash", str(SUPERVISOR)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Uso:" in result.stdout


def test_supervisor_retries_after_kubectl_exit(
    tmp_path: Path,
) -> None:
    attempts_file = tmp_path / "attempts.log"

    bin_dir = _write_fake_kubectl(
        tmp_path,
        (
            'echo "$*" >> "${ATTEMPTS_FILE}"\n'
            "exit 1"
        ),
    )

    env = _supervisor_env(bin_dir)
    env["ATTEMPTS_FILE"] = str(attempts_file)

    process = subprocess.Popen(
        [
            "bash",
            str(SUPERVISOR),
            "mlflow",
            "mlflow",
            "5000:5000",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_until(
            lambda: (
                attempts_file.exists()
                and len(
                    attempts_file.read_text(
                        encoding="utf-8"
                    ).splitlines()
                )
                >= 2
            ),
            timeout=6.0,
        )
    finally:
        process.terminate()
        process.wait(timeout=5)

    attempts = attempts_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(attempts) >= 2
    assert all(
        attempt
        == "port-forward svc/mlflow 5000:5000"
        for attempt in attempts
    )


@pytest.mark.skipif(
    os.name != "posix",
    reason="Teste depende de sinais POSIX.",
)
def test_supervisor_terminates_child_on_sigterm(
    tmp_path: Path,
) -> None:
    child_started = tmp_path / "child-started.txt"
    child_terminated = (
        tmp_path / "child-terminated.txt"
    )

    bin_dir = _write_fake_kubectl(
        tmp_path,
        (
            'echo "started" > "${CHILD_STARTED}"\n'
            "\n"
            "terminate_child() {\n"
            '    echo "terminated" '
            '> "${CHILD_TERMINATED}"\n'
            "    exit 0\n"
            "}\n"
            "\n"
            "trap terminate_child TERM INT\n"
            "\n"
            "while true; do\n"
            "    sleep 1\n"
            "done"
        ),
    )

    env = _supervisor_env(bin_dir)
    env["CHILD_STARTED"] = str(child_started)
    env["CHILD_TERMINATED"] = str(
        child_terminated
    )

    process = subprocess.Popen(
        [
            "bash",
            str(SUPERVISOR),
            "mlflow",
            "mlflow",
            "5000:5000",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_until(
            child_started.exists,
            timeout=3.0,
        )

        process.terminate()
        process.wait(timeout=5)

        _wait_until(
            child_terminated.exists,
            timeout=3.0,
        )

        assert process.returncode == 0

    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
