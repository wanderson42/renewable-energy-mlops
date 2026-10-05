"""Check stage persistence, target isolation and paired inference gates without a cluster."""

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("repro_serving", ROOT / "scripts/repro-serving.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def metadata():
    return {"model_name": MODULE.MODEL, "alias": "champion", "version": 17,
            "run_id": MODULE.RUN, "metrics": {"oot_mae_mw": 787.1835861175788}}


def predictions():
    return [{"date": row["date"], "predicted_fc": 0.2 + index / 10,
             "predicted_mw": 2000.0 + index * 100} for index, row in enumerate(MODULE.PAYLOAD["predictions"])]


def plan():
    before = {"name": "energy-mlops-repro", "namespace": "energy-mlops-repro"}
    return {"variables": {"api_enabled": {"value": True}}, "resource_changes": [
        {"address": "helm_release.mlops", "change": {"actions": ["update"],
         "before": before, "after": {**before, "set": [{"name": "api.enabled", "value": "true"}]}}}
    ]}


class ServingGateTests(unittest.TestCase):
    def test_paired_batch_accepts_equal_predictions_and_equivalent_utc_dates(self):
        destination = predictions()
        destination[0]["date"] = destination[0]["date"].replace("Z", "+00:00")
        self.assertEqual(MODULE.compare_predictions(predictions(), destination), {"predicted_fc": 0.0, "predicted_mw": 0.0})

    def test_batch_rejects_wrong_length_dates_ranges_and_nonfinite_outputs(self):
        cases = [predictions()[:-1]]
        for field, value in (("date", "2026-10-04T13:00:00Z"), ("predicted_fc", 1.1),
                             ("predicted_mw", None), ("predicted_mw", float("nan")),
                             ("predicted_mw", float("inf")), ("predicted_fc", -0.1),
                             ("predicted_fc", True)):
            changed = predictions()
            changed[0][field] = value
            cases.append(changed)
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                MODULE.compare_predictions(predictions(), changed)

    def test_batch_rejects_inference_differences(self):
        changed = predictions()
        changed[1]["predicted_mw"] += 0.01
        with self.assertRaises(RuntimeError):
            MODULE.compare_predictions(predictions(), changed)

    def test_endpoint_checks_identity_health_metrics_and_no_mutation(self):
        calls = []

        def respond(base, path, payload=None):
            calls.append((base, path, payload))
            return {"/health": {"status": "healthy", "model_loaded": True},
                    "/model-info": metadata(), "/predict/batch": predictions()}[path]

        with patch.object(MODULE, "request", side_effect=respond):
            result = MODULE.compare_endpoints("http://127.0.0.1:34567")
        self.assertEqual(result["reference"], "paired_synthetic_batch_v1")
        self.assertEqual([payload for _, path, payload in calls if path == "/predict/batch"], [MODULE.PAYLOAD, MODULE.PAYLOAD])
        self.assertFalse(any(path == "/reload-model" for _, path, _ in calls))
        bad = metadata()
        bad["version"] = 10
        with self.assertRaises(RuntimeError):
            MODULE.check_identity(bad)

    def test_health_failure_and_alias_change_during_comparison_are_rejected(self):
        for failure in ("health", "changed_alias"):
            counts = {}

            def respond(base, path, payload=None):
                counts[(base, path)] = counts.get((base, path), 0) + 1
                if path == "/health":
                    return {"status": "healthy", "model_loaded": failure != "health"}
                if path == "/predict/batch":
                    return predictions()
                info = metadata()
                if failure == "changed_alias" and counts[(base, path)] > 1:
                    info["run_id"] = "changed"
                return info

            with self.subTest(failure=failure), patch.object(MODULE, "request", side_effect=respond), self.assertRaises(RuntimeError):
                MODULE.compare_endpoints("http://127.0.0.1:34567")


class ServingDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / ".repro").mkdir(mode=0o700)
        self.root_patch = patch.object(MODULE, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def test_plan_rejects_install_destroy_replacement_and_wrong_namespace(self):
        MODULE.inspect_plan(plan())
        for actions in (["create"], ["delete"], ["delete", "create"]):
            changed = plan()
            changed["resource_changes"][0]["change"]["actions"] = actions
            with self.subTest(actions=actions), self.assertRaises(RuntimeError):
                MODULE.inspect_plan(changed)
        changed = plan()
        changed["resource_changes"][0]["change"]["after"]["namespace"] = "default"
        with self.assertRaises(RuntimeError):
            MODULE.inspect_plan(changed)

    def test_wrong_node_or_changed_live_registry_blocks_enablement(self):
        with patch.object(MODULE, "run", return_value="energy-mlops-control-plane"), self.assertRaises(RuntimeError):
            MODULE.check_target()
        model = {"name": MODULE.MODEL, "version": 17, "run_id": MODULE.RUN, "source": "models:/logged", "download_uri": "s3://mlflow-artifacts/model"}
        receipt = {"model": model, "verified_objects": 251, "verified_bytes": 1829741640}
        (self.root / ".repro/restored-data.json").write_text(json.dumps(receipt))
        with patch.object(MODULE, "run", return_value=json.dumps({**model, "version": 18})), self.assertRaises(RuntimeError):
            MODULE.check_restore()

    def test_apply_saves_intent_and_runs_gate_only_after_successful_terraform(self):
        calls = []

        def execute(args, **kwargs):
            calls.append(args)
            return json.dumps(plan()) if "show" in args else None

        with patch.object(MODULE, "check_target"), patch.object(MODULE, "check_restore"), \
             patch.object(MODULE, "run", side_effect=execute), patch.object(MODULE, "validate_serving") as gate:
            MODULE.serving_apply()
        self.assertEqual(json.loads((self.root / ".repro/deployment.tfvars.json").read_text()), {"api_enabled": True})
        self.assertEqual((self.root / ".repro/deployment.tfvars.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual(calls[-1], MODULE.terraform("apply", "-input=false", str(self.root / ".repro/serving.tfplan")))
        gate.assert_called_once()

    def test_failed_apply_preserves_serving_intent_and_does_not_report_gate_success(self):
        def execute(args, **kwargs):
            if "show" in args:
                return json.dumps(plan())
            raise subprocess.CalledProcessError(1, args)

        with patch.object(MODULE, "check_target"), patch.object(MODULE, "check_restore"), \
             patch.object(MODULE, "run", side_effect=execute), patch.object(MODULE, "validate_serving") as gate, \
             self.assertRaises(subprocess.CalledProcessError):
            MODULE.serving_apply()
        self.assertTrue((self.root / ".repro/deployment.tfvars.json").exists())
        gate.assert_not_called()

    def test_validation_terminates_only_its_own_forward_on_gate_failure(self):
        class Process:
            terminated = False

            def poll(self):
                return 0 if self.terminated else None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout):
                return 0

        process = Process()
        captured = []

        def start(args, stdout, **kwargs):
            captured.append(args)
            stdout.write("Forwarding from 127.0.0.1:34567 -> 8000\n")
            stdout.flush()
            return process

        def execute(args, **kwargs):
            if "deployment" in args:
                return json.dumps({"spec": {"template": {"spec": {"containers": [{"image": MODULE.IMAGE}]}}}})
            if "pods" in args:
                return json.dumps({"items": [{"metadata": {}, "status": {"containerStatuses": [{"ready": True, "imageID": MODULE.IMAGE}]}}]})

        with patch.object(MODULE, "check_target"), patch.object(MODULE, "run", side_effect=execute), \
             patch.object(MODULE.subprocess, "Popen", side_effect=start), \
             patch.object(MODULE, "compare_endpoints", side_effect=RuntimeError("divergence")), self.assertRaises(RuntimeError):
            MODULE.validate_serving()
        self.assertTrue(process.terminated)
        self.assertEqual(captured[0], MODULE.kubectl("port-forward", "--address=127.0.0.1", "service/energy-api", ":8000"))
        self.assertFalse((self.root / ".repro/serving-validation.json").exists())


class PlanWrapperTests(unittest.TestCase):
    def test_later_plans_preserve_serving_and_serving_plan_does_not_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("scripts", ".repro/bin", "helm", "terraform/environments/local", "bin"):
                (root / name).mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / "scripts/plan-repro.sh", root / "scripts/plan-repro.sh")
            (root / ".repro/kubeconfig").touch()
            (root / "helm/values_secrets.yaml").write_text("{}\n")
            (root / ".repro/deployment.tfvars.json").write_text('{"api_enabled": true}\n')
            terraform = root / ".repro/bin/terraform"
            terraform.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with Path(os.environ["CALL_LOG"]).open("a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
''')
            terraform.chmod(0o755)
            kubectl = root / "bin/kubectl"
            kubectl.write_text('#!/bin/sh\nprintf "%s" "${FAKE_NODE:-energy-mlops-repro-control-plane}"\n')
            kubectl.chmod(0o755)
            environment = {**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                           "CALL_LOG": str(root / "calls.jsonl")}
            for args in ([], ["--serving"]):
                result = subprocess.run(["bash", str(root / "scripts/plan-repro.sh"), *args],
                                        env=environment, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            plans = [args for args in calls if "plan" in args]
            self.assertEqual(len(plans), 2)
            self.assertTrue(all(f"-var-file={root / '.repro/deployment.tfvars.json'}" in args for args in plans))
            self.assertIn("-var=api_enabled=true", plans[1])
            self.assertIn(f"-out={root / '.repro/serving.tfplan'}", plans[1])
            self.assertFalse(any("apply" in args for args in calls))
            count = len(calls)
            result = subprocess.run(["bash", str(root / "scripts/plan-repro.sh")],
                                    env={**environment, "FAKE_NODE": "energy-mlops-control-plane"},
                                    text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len((root / "calls.jsonl").read_text().splitlines()), count)


if __name__ == "__main__":
    unittest.main()
