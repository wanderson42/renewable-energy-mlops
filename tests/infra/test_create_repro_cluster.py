"""Test bootstrap isolation with fake CLIs; never create a real cluster."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAKE_CLI = '''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
with open(os.environ["CALLS_LOG"], "a") as stream:
    stream.write(json.dumps([name, *sys.argv[1:]]) + "\\n")
if name == "kind" and sys.argv[1:] == ["get", "clusters"]:
    print(os.environ.get("FAKE_CLUSTERS", "energy-mlops"))
elif name == "kind" and sys.argv[1:3] == ["create", "cluster"]:
    path = sys.argv[sys.argv.index("--kubeconfig") + 1]
    Path(path).write_text("test kubeconfig")
'''


class ReproClusterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "infra/kind").mkdir(parents=True)
        self.script = self.root / "scripts/create-repro-cluster.sh"
        shutil.copyfile(
            PROJECT_ROOT / "scripts/create-repro-cluster.sh", self.script
        )
        shutil.copyfile(
            PROJECT_ROOT / "infra/kind/repro.yaml",
            self.root / "infra/kind/repro.yaml",
        )
        self.bin_dir = Path(self.temp.name) / "bin"
        self.bin_dir.mkdir()
        for tool in ("docker", "kind", "kubectl"):
            path = self.bin_dir / tool
            path.write_text(FAKE_CLI, encoding="utf-8")
            path.chmod(0o755)
        self.log = Path(self.temp.name) / "calls.jsonl"
        self.operational_config = Path(self.temp.name) / "operational-config"
        self.operational_config.write_text("operational context stays here")
        self.env = {
            **os.environ,
            "PATH": f"{self.bin_dir}{os.pathsep}{os.environ['PATH']}",
            "CALLS_LOG": str(self.log),
            "KUBECONFIG": str(self.operational_config),
            "KIND_EXPERIMENTAL_PROVIDER": "podman",
        }

    def run_bootstrap(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.script)],
            cwd=self.temp.name,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    def calls(self) -> list[list[str]]:
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_creation_is_scoped_and_preserves_operational_kubeconfig(self) -> None:
        result = self.run_bootstrap()
        self.assertEqual(result.returncode, 0, result.stderr)
        kubeconfig = self.root / ".repro/kubeconfig"
        calls = self.calls()
        create = [call for call in calls if call[:3] == ["kind", "create", "cluster"]]
        self.assertEqual(create, [[
            "kind", "create", "cluster",
            "--name", "energy-mlops-repro",
            "--config", str(self.root / "infra/kind/repro.yaml"),
            "--kubeconfig", str(kubeconfig),
            "--wait", "120s",
        ]])
        for call in calls:
            if call[0] == "kubectl":
                self.assertEqual(call[1:5], [
                    "--kubeconfig", str(kubeconfig),
                    "--context", "kind-energy-mlops-repro",
                ])
        self.assertEqual(
            self.operational_config.read_text(), "operational context stays here"
        )
        self.assertEqual(kubeconfig.stat().st_mode & 0o777, 0o600)
        self.assertEqual(kubeconfig.parent.stat().st_mode & 0o777, 0o700)
        self.assertFalse(any("delete" in call for call in calls))

    def test_existing_cluster_is_not_recreated(self) -> None:
        self.env["FAKE_CLUSTERS"] = "energy-mlops\nenergy-mlops-repro"
        result = self.run_bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("já existe", result.stderr)
        self.assertFalse(any("create" in call for call in self.calls()))

    def test_existing_kubeconfig_is_not_overwritten(self) -> None:
        kubeconfig = self.root / ".repro/kubeconfig"
        kubeconfig.parent.mkdir()
        kubeconfig.write_text("existing kubeconfig")
        result = self.run_bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("não será sobrescrito", result.stderr)
        self.assertEqual(kubeconfig.read_text(), "existing kubeconfig")
        self.assertFalse(any("create" in call for call in self.calls()))

    def test_node_image_matches_operational_baseline(self) -> None:
        baseline = json.loads(
            (PROJECT_ROOT / "docs/evidence/deployment_baseline_2026-10-05.json")
            .read_text(encoding="utf-8")
        )
        config = (self.root / "infra/kind/repro.yaml").read_text()
        self.assertIn(f"image: {baseline['node_image']}", config)
        self.assertNotIn("extraMounts:", config)
        self.assertNotIn("extraPortMappings:", config)


if __name__ == "__main__":
    unittest.main()
