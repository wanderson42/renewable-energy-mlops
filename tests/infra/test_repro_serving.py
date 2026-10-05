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
REVIEW_SPEC = importlib.util.spec_from_file_location("review_repro_plan", ROOT / "scripts/review-repro-plan.py")
REVIEW = importlib.util.module_from_spec(REVIEW_SPEC)
REVIEW_SPEC.loader.exec_module(REVIEW)
TERRAFORM = shutil.which("terraform")
if not TERRAFORM and os.access(ROOT / ".repro/bin/terraform", os.X_OK):
    TERRAFORM = str(ROOT / ".repro/bin/terraform")


def metadata():
    return {"model_name": MODULE.MODEL, "alias": "champion", "version": 17,
            "run_id": MODULE.RUN, "metrics": {"oot_mae_mw": 787.1835861175788}}


def predictions():
    return [{"date": row["date"], "predicted_fc": 0.2 + index / 10,
             "predicted_mw": 2000.0 + index * 100} for index, row in enumerate(MODULE.PAYLOAD["predictions"])]


def plan():
    before = {"name": "energy-mlops-repro", "namespace": "energy-mlops-repro"}
    target = {"context": "kind-energy-mlops-repro", "namespace": "energy-mlops-repro",
              "release": "energy-mlops-repro", "api_enabled": True,
              "api_digest": MODULE.DEFAULT_DIGEST, "deployment_timeout_seconds": 600}
    before["set"] = [{"name": "api.enabled", "value": "true"}, {"name": "repro.chartHash", "value": "a" * 64}]
    return {"variables": {"api_enabled": {"value": "true"}},
            "planned_values": {"outputs": {"deployment_target": {"value": target}}}, "resource_changes": [
        {"address": "helm_release.mlops", "change": {"actions": ["update"],
         "before": before, "after": {**before, "timeout": 600,
         "set": [*[dict(item) for item in before["set"]], {"name": "api.image.digest", "value": MODULE.DEFAULT_DIGEST}]}}}
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

    def test_plan_checks_evaluated_boolean_output_instead_of_raw_cli_string(self):
        MODULE.inspect_plan(plan())
        for value in (False, "true", None):
            changed = plan()
            changed["planned_values"]["outputs"]["deployment_target"]["value"]["api_enabled"] = value
            with self.subTest(value=value), self.assertRaises(RuntimeError):
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
        self.assertEqual(json.loads((self.root / ".repro/deployment.tfvars.json").read_text()), MODULE.configuration({}))
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


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / ".repro").mkdir(mode=0o700)
        self.root_patch = patch.object(MODULE, "ROOT", self.root)
        self.root_patch.start()
        self.volumes = {"postgres-pvc": "postgres-uid", "rustfs-pvc": "rustfs-uid"}
        self.result = {"model": metadata(), "image": MODULE.IMAGE, "validated_at": "2026-10-05T21:25:47Z",
                       "max_absolute_difference": {"predicted_fc": 0.0, "predicted_mw": 0.0}}
        self.saved = {**self.result, "configuration": MODULE.configuration({}), "pvc_identity": self.volumes}

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def test_checkpoint_requires_successful_gate_and_preserves_bound_pvc_identity(self):
        with patch.object(MODULE, "validate_serving", side_effect=RuntimeError("unhealthy")), self.assertRaises(RuntimeError):
            MODULE.checkpoint()
        self.assertFalse((self.root / ".repro/rollback-checkpoint.json").exists())
        with patch.object(MODULE, "validate_serving", return_value=self.result), patch.object(MODULE, "volume_identity", return_value=self.volumes):
            saved = MODULE.checkpoint()
        self.assertEqual(saved["pvc_identity"], self.volumes)
        self.assertEqual((self.root / ".repro/rollback-checkpoint.json").stat().st_mode & 0o777, 0o600)

    def test_mismatched_checkpoint_volumes_prevent_all_rollback_mutations(self):
        MODULE.write_private(self.root / ".repro/rollback-checkpoint.json", self.saved)
        with patch.object(MODULE, "check_target"), patch.object(MODULE, "volume_identity", return_value={}), \
             patch.object(MODULE, "serving_plan") as prepare, self.assertRaises(RuntimeError):
            MODULE.rollback()
        prepare.assert_not_called()
        self.assertFalse((self.root / ".repro/deployment.tfvars.json").exists())

    def test_recovery_rejects_chart_credentials_and_non_digest_setting_changes(self):
        MODULE.require_image_only_change(plan())
        for field in ("version", "values", "chart"):
            changed = plan()
            changed["resource_changes"][0]["change"]["after"][field] = "different"
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                MODULE.require_image_only_change(changed)
        changed = plan()
        changed["resource_changes"][0]["change"]["after"]["set"][1]["value"] = "b" * 64
        with self.assertRaises(RuntimeError):
            MODULE.require_image_only_change(changed)

    def test_digest_and_wait_deadline_must_match_evaluated_output(self):
        for field, value in (("api_digest", "sha256:" + "b" * 64), ("deployment_timeout_seconds", 60)):
            changed = plan()
            changed["planned_values"]["outputs"]["deployment_target"]["value"][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                MODULE.inspect_plan(changed)

    def test_rollback_reapplies_checkpoint_and_handles_failed_status_with_unchanged_inputs(self):
        MODULE.write_private(self.root / ".repro/rollback-checkpoint.json", self.saved)
        MODULE.write_private(self.root / ".repro/deployment.tfvars.json", {"api_digest": MODULE.FAILURE_DIGEST, "deployment_timeout_seconds": 60})
        update = plan()
        update["resource_changes"][0]["change"]["before"]["status"] = "failed"
        update["resource_changes"][0]["change"]["after"]["status"] = "deployed"
        with patch.object(MODULE, "check_target"), patch.object(MODULE, "volume_identity", return_value=self.volumes), \
             patch.object(MODULE, "serving_plan", return_value=update), patch.object(MODULE, "serving_apply", return_value=self.result) as apply:
            restored = MODULE.rollback()
        self.assertEqual(restored["image"], MODULE.IMAGE)
        self.assertEqual(json.loads((self.root / ".repro/deployment.tfvars.json").read_text()), self.saved["configuration"])
        apply.assert_called_once()

    def exercise_recovery(self, reason):
        pods = {"items": [{"status": {"containerStatuses": [{"state": {"waiting": {"reason": reason}}}]}}]}
        unchanged = plan()
        unchanged["resource_changes"][0]["change"]["actions"] = ["no-op"]

        def execute(args, **kwargs):
            return json.dumps(pods if "pods" in args else unchanged)

        with patch.object(MODULE, "checkpoint"), patch.object(MODULE, "failure_plan"), \
             patch.object(MODULE, "serving_apply", side_effect=subprocess.CalledProcessError(1, ["terraform", "apply"])), \
             patch.object(MODULE, "rollback", return_value=self.result) as rollback, patch.object(MODULE, "run", side_effect=execute):
            try:
                MODULE.recovery_test()
            finally:
                rollback.assert_called_once()

    def test_controlled_pull_failure_is_recovered_and_evidence_requires_no_change_plan(self):
        (self.root / ".repro/terraform-apply.log").write_text("FAKE_PRIVATE_DIAGNOSTIC")
        self.exercise_recovery("ImagePullBackOff")
        evidence = json.loads((self.root / ".repro/recovery-validation.json").read_text())
        self.assertTrue(evidence["expected_pull_failure"])
        self.assertEqual(evidence["post_recovery_plan"], "No changes")
        self.assertEqual((self.root / ".repro/recovery-failed-apply.log").read_text(), "FAKE_PRIVATE_DIAGNOSTIC")
        self.assertNotIn("FAKE_PRIVATE_DIAGNOSTIC", json.dumps(evidence))

    def test_unexpected_failure_still_runs_rollback_and_never_reports_success(self):
        with self.assertRaises(RuntimeError):
            self.exercise_recovery("CrashLoopBackOff")
        self.assertFalse((self.root / ".repro/recovery-validation.json").exists())


class PlanWrapperTests(unittest.TestCase):
    def test_later_plans_preserve_serving_and_serving_plan_does_not_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("scripts", ".repro/bin", "helm", "terraform/environments/local", "bin"):
                (root / name).mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / "scripts/plan-repro.sh", root / "scripts/plan-repro.sh")
            shutil.copy(ROOT / "scripts/review-repro-plan.py", root / "scripts/review-repro-plan.py")
            (root / ".repro/kubeconfig").touch()
            (root / "helm/values_secrets.yaml").write_text("{}\n")
            (root / ".repro/deployment.tfvars.json").write_text('{"api_enabled": true}\n')
            terraform = root / ".repro/bin/terraform"
            terraform.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with Path(os.environ["CALL_LOG"]).open("a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
if "plan" in sys.argv:
    print("RAW_FAKE_SECRET_KEEP_PRIVATE")
    if os.environ.get("PLAN_FAIL"):
        raise SystemExit(2)
elif "show" in sys.argv:
    print(os.environ["FAKE_PLAN_JSON"])
''')
            terraform.chmod(0o755)
            kubectl = root / "bin/kubectl"
            kubectl.write_text('#!/bin/sh\nprintf "%s" "${FAKE_NODE:-energy-mlops-repro-control-plane}"\n')
            kubectl.chmod(0o755)
            environment = {**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                           "CALL_LOG": str(root / "calls.jsonl"), "FAKE_PLAN_JSON": json.dumps(plan())}
            for args in ([], ["--serving"], ["--serving", MODULE.FAILURE_DIGEST, "60"]):
                result = subprocess.run(["bash", str(root / "scripts/plan-repro.sh"), *args],
                                        env=environment, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("RAW_FAKE_SECRET_KEEP_PRIVATE", result.stdout + result.stderr)
                self.assertIn("RAW_FAKE_SECRET_KEEP_PRIVATE", (root / ".repro/terraform-plan.log").read_text())
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            plans = [args for args in calls if "plan" in args]
            self.assertEqual(len(plans), 3)
            self.assertTrue(all(f"-var-file={root / '.repro/deployment.tfvars.json'}" in args for args in plans))
            self.assertIn("-var=api_enabled=true", plans[1])
            self.assertIn(f"-out={root / '.repro/serving.tfplan'}", plans[1])
            self.assertIn(f"-var=api_digest={MODULE.FAILURE_DIGEST}", plans[2])
            self.assertIn("-var=deployment_timeout_seconds=60", plans[2])
            self.assertFalse(any("apply" in args for args in calls))
            count = len(calls)
            result = subprocess.run(["bash", str(root / "scripts/plan-repro.sh")],
                                    env={**environment, "FAKE_NODE": "energy-mlops-control-plane"},
                                    text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len((root / "calls.jsonl").read_text().splitlines()), count)
            result = subprocess.run(["bash", str(root / "scripts/plan-repro.sh"), "--serving"],
                                    env={**environment, "PLAN_FAIL": "1"}, text=True,
                                    capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("RAW_FAKE_SECRET_KEEP_PRIVATE", result.stdout + result.stderr)
            self.assertIn("RAW_FAKE_SECRET_KEEP_PRIVATE", (root / ".repro/terraform-plan.log").read_text())
            self.assertEqual((root / ".repro/terraform-plan.log").stat().st_mode & 0o777, 0o600)


class PlanReviewTests(unittest.TestCase):
    def test_review_omits_secrets_even_in_provider_metadata_and_unexpected_settings(self):
        raw = plan()
        raw["variables"]["password"] = {"value": "FAKE_PRIVATE_INPUT"}
        for side in ("before", "after"):
            resource = raw["resource_changes"][0]["change"][side]
            resource["metadata"] = {"values": '{"password":"FAKE_PRIVATE_METADATA"}'}
            resource["values"] = ["password: FAKE_PRIVATE_VALUES"]
            resource.setdefault("set", []).append({"name": "postgres.password", "value": "FAKE_PRIVATE_SET"})
        serialized = json.dumps(REVIEW.review(raw))
        self.assertNotIn("FAKE_PRIVATE", serialized)
        self.assertIn('"planned_api_enabled": true', serialized)
        self.assertEqual(REVIEW.review(raw)["summary"], {"add": 0, "change": 1, "destroy": 0})

    def test_no_change_summary_and_output_only_changes(self):
        raw = plan()
        raw["resource_changes"][0]["change"]["actions"] = ["no-op"]
        self.assertTrue(REVIEW.review(raw)["no_changes"])
        raw["output_changes"] = {"deployment_target": {"actions": ["update"]}}
        self.assertFalse(REVIEW.review(raw)["no_changes"])

    @unittest.skipUnless(TERRAFORM, "Terraform CLI needed for provider-free regression")
    def test_real_terraform_evaluates_cli_and_json_var_file_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.tf").write_text('''variable "api_enabled" {
  type = bool
  default = false
}
output "deployment_target" {
  value = {
    context = "kind-energy-mlops-repro"
    namespace = "energy-mlops-repro"
    release = "energy-mlops-repro"
    api_enabled = var.api_enabled
    api_digest = "sha256:95208ab282e24014a81f60d1e3eac01b8db36e40f05ae25e9f168264e4b7c188"
    deployment_timeout_seconds = 600
  }
}
''')
            (root / "enabled.tfvars.json").write_text('{"api_enabled": true}\n')
            subprocess.run([TERRAFORM, f"-chdir={root}", "init", "-input=false"],
                           capture_output=True, text=True, check=True, timeout=30)
            for argument in ("-var=api_enabled=true", "-var-file=enabled.tfvars.json"):
                with self.subTest(argument=argument):
                    subprocess.run([TERRAFORM, f"-chdir={root}", "plan", "-input=false", argument, "-out=plan"],
                                   capture_output=True, text=True, check=True, timeout=30)
                    result = subprocess.run([TERRAFORM, f"-chdir={root}", "show", "-json", "plan"],
                                            capture_output=True, text=True, check=True, timeout=30)
                    evaluated = json.loads(result.stdout)
                    # Helm provider unavailable here: attach an isolated update fixture.
                    evaluated["resource_changes"] = plan()["resource_changes"]
                    MODULE.inspect_plan(evaluated)
                    self.assertIs(evaluated["planned_values"]["outputs"]["deployment_target"]["value"]["api_enabled"], True)


if __name__ == "__main__":
    unittest.main()
