"""Render the chart to protect image identity without contacting a cluster.

Standalone execution needs only Python and Helm:
python3 -m unittest discover -s tests/infra -p test_helm_chart.py -v
"""

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELM = shutil.which("helm")


@unittest.skipUnless(HELM, "Helm is required for chart rendering tests")
class HelmImageTests(unittest.TestCase):
    def render(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [HELM, "template", "energy-mlops-repro", "helm", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def test_default_images_preserve_existing_references(self) -> None:
        result = self.render()
        self.assertEqual(result.returncode, 0, result.stderr)
        images = re.findall(r'^\s+image: "([^"]+)"$', result.stdout, re.M)
        self.assertCountEqual(
            images,
            [
                "ghcr.io/wanderson42/renewable-energy-mlops:latest",
                "ghcr.io/mlflow/mlflow:v3.16.1-full",
                "postgres:13",
                "rustfs/rustfs:latest",
            ],
        )
        policies = re.findall(
            r'^\s+imagePullPolicy: "([^"]+)"$', result.stdout, re.M
        )
        self.assertCountEqual(
            policies, ["Always", "Always", "IfNotPresent", "IfNotPresent"]
        )

    def test_repro_images_match_baseline_and_published_mlflow_runtime(self) -> None:
        baseline_path = (
            PROJECT_ROOT / "docs/evidence/deployment_baseline_2026-10-05.json"
        )
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        for workload, template in (
            ("energy-api", "api"),
            ("mlflow", "mlflow"),
            ("postgres", "postgres"),
            ("rustfs", "rustfs"),
        ):
            with self.subTest(workload=workload):
                result = self.render(
                    "-f", "helm/environments/repro.yaml",
                    "--show-only", f"templates/{template}.yaml",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                images = re.findall(
                    r'^\s+image: "([^"]+)"$', result.stdout, re.M
                )
                expected = baseline["workloads"][workload]["image_id"]
                if workload == "mlflow":
                    evidence = json.loads((PROJECT_ROOT / "docs/evidence/mlflow_runtime_build_2026-10-05.json").read_text())
                    expected = evidence["published_image"]["reference"]
                self.assertEqual(images, [expected])
                policies = re.findall(
                    r'^\s+imagePullPolicy: "([^"]+)"$', result.stdout, re.M
                )
                self.assertEqual(policies, ["IfNotPresent"])

    def test_digest_overrides_tag_and_pull_policy_is_configurable(self) -> None:
        digest = "sha256:" + "a" * 64
        result = self.render(
            "--show-only",
            "templates/api.yaml",
            "--set-string",
            f"api.image.digest={digest}",
            "--set-string",
            "api.image.tag=must-not-be-used",
            "--set-string",
            "api.image.pullPolicy=Never",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f'image: "ghcr.io/wanderson42/renewable-energy-mlops@{digest}"',
            result.stdout,
        )
        self.assertNotIn("must-not-be-used", result.stdout)
        self.assertIn('imagePullPolicy: "Never"', result.stdout)

    def test_bootstrap_waits_for_model_restore_before_serving(self) -> None:
        result = self.render(
            "-f", "helm/environments/repro.yaml",
            "-f", "helm/environments/repro-bootstrap.yaml",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("name: energy-api", result.stdout)
        images = re.findall(r'^\s+image: "([^"]+)"$', result.stdout, re.M)
        self.assertEqual(len(images), 3)
        for component in ("mlflow", "postgres", "rustfs"):
            self.assertIn(f"name: {component}", result.stdout)

    def test_invalid_digest_fails_for_every_component(self) -> None:
        for component in ("api", "mlflow", "postgres", "rustfs"):
            with self.subTest(component=component):
                result = self.render(
                    "--set-string", f"{component}.image.digest=sha256:incomplete"
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("image.digest must be sha256", result.stderr)

    def test_mlflow_runtime_matches_observed_server_packages(self) -> None:
        baseline = json.loads((PROJECT_ROOT / "docs/evidence/deployment_baseline_2026-10-05.json").read_text())
        result = self.render("--show-only", "templates/mlflow.yaml")
        self.assertEqual(result.returncode, 0, result.stderr)
        packages = re.findall(r"'([a-z0-9-]+)==([^']+)'", result.stdout)
        self.assertEqual(dict(packages), baseline["workloads"]["mlflow"]["runtime"]["packages"])

    def test_packaged_mlflow_starts_without_installing_packages(self) -> None:
        result = self.render("-f", "helm/environments/repro.yaml", "--show-only", "templates/mlflow.yaml")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("pip install", result.stdout)
        self.assertIn("exec mlflow server", result.stdout)
        invalid = self.render("--set-string", "mlflow.installRuntimePackages=false")
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("must be a boolean", invalid.stderr)

    def test_bootstrap_readiness_checks_do_not_depend_on_champion(self) -> None:
        for component, expected in (("mlflow", "port: 5000"), ("rustfs", "port: 9000"), ("postgres", "pg_isready")):
            with self.subTest(component=component):
                result = self.render("--show-only", f"templates/{component}.yaml")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("readinessProbe:", result.stdout)
                self.assertIn(expected, result.stdout)
                self.assertNotIn("/model-info", result.stdout)

    def test_image_without_tag_or_digest_is_rejected(self) -> None:
        for component in ("api", "mlflow", "postgres", "rustfs"):
            with self.subTest(component=component):
                result = self.render("--set-string", f"{component}.image.tag=")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("image.tag is required", result.stderr)

    def test_api_waits_for_a_loaded_model_before_becoming_ready(self) -> None:
        result = self.render("--show-only", "templates/api.yaml")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("startupProbe:", result.stdout)
        self.assertIn("readinessProbe:", result.stdout)
        self.assertEqual(result.stdout.count("path: /health"), 2)


if __name__ == "__main__":
    unittest.main()
