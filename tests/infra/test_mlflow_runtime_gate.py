"""Reject image, runtime and artifact mismatches without contacting a cluster."""

import copy
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("mlflow_runtime_gate", ROOT / "scripts/validate-mlflow-runtime.py")
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class RuntimeGateTests(unittest.TestCase):
    def test_deployment_requires_ready_pinned_image_and_no_startup_install(self):
        deployment = {"spec": {"template": {"spec": {"containers": [
            {"name": "mlflow", "image": GATE.IMAGE, "command": ["sh", "-c", "exec mlflow server"]}
        ]}}}}
        pods = {"items": [{"spec": {"containers": [{"name": "mlflow", "image": GATE.IMAGE}]},
                           "status": {"containerStatuses": [{"name": "mlflow", "ready": True, "imageID": GATE.IMAGE}]}}]}
        self.assertEqual(GATE.check_deployment(deployment, pods), GATE.IMAGE)
        changed = copy.deepcopy(deployment)
        changed["spec"]["template"]["spec"]["containers"][0]["command"] = ["sh", "-c", "pip install boto3 && mlflow server"]
        with self.assertRaises(RuntimeError):
            GATE.check_deployment(changed, pods)
        pods["items"][0]["status"]["containerStatuses"][0]["ready"] = False
        with self.assertRaises(RuntimeError):
            GATE.check_deployment(deployment, pods)
        changed = copy.deepcopy(deployment)
        changed["spec"]["template"]["spec"]["containers"][0]["image"] = "wrong:tag"
        with self.assertRaises(RuntimeError):
            GATE.check_deployment(changed, pods)

    def test_runtime_rejects_wrong_python_versions_and_incomplete_checks(self):
        GATE.check_runtime(GATE.BUILD["runtime"])
        for field, value in (("python", "3.14.0"), ("packages", {}), ("pip_check_passed", False), ("imports_passed", False)):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                GATE.check_runtime({**GATE.BUILD["runtime"], field: value})

    def test_artifact_requires_unique_bounded_metadata_with_valid_hash(self):
        manifest = {"model": {"download_uri": "s3://mlflow-artifacts/6/model/artifacts"},
                    "objects": [{"bucket": "mlflow-artifacts", "key": "6/model/artifacts/MLmodel", "size": 123, "sha256": "a" * 64}]}
        self.assertEqual(GATE.expected_artifact(manifest)["bytes"], 123)
        for records in ([], manifest["objects"] * 2, [{**manifest["objects"][0], "size": 1048577}], [{**manifest["objects"][0], "sha256": "invalid"}]):
            with self.subTest(records=records), self.assertRaises(RuntimeError):
                GATE.expected_artifact({**manifest, "objects": records})
        with self.assertRaises(RuntimeError):
            GATE.expected_artifact({**manifest, "model": {"download_uri": "s3://another-bucket/model"}})

    def test_full_gate_rejects_corrupt_artifact_or_changed_pvcs_before_writing_success(self):
        model = {"download_uri": "s3://mlflow-artifacts/6/model/artifacts"}
        manifest = {"model": model, "objects": [{"bucket": "mlflow-artifacts", "key": "6/model/artifacts/MLmodel", "size": 123, "sha256": "a" * 64}]}
        artifact = GATE.expected_artifact(manifest)
        deployment = {"spec": {"template": {"spec": {"containers": [{"name": "mlflow", "image": GATE.IMAGE}]}}}}
        pods = {"items": [{"spec": {"containers": [{"name": "mlflow", "image": GATE.IMAGE}]}, "status": {
            "containerStatuses": [{"name": "mlflow", "ready": True, "imageID": GATE.IMAGE}]}}]}
        for actual_artifact, volumes, succeeds in ((artifact, {"uid": "same"}, True),
                ({**artifact, "sha256": "b" * 64}, {"uid": "same"}, False),
                (artifact, {"uid": "changed"}, False)):
            with self.subTest(succeeds=succeeds, volumes=volumes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / ".repro").mkdir()
                (root / ".repro/restore-manifest.json").write_text(json.dumps(manifest))
                (root / ".repro/restored-data.json").write_text(json.dumps({"model": model}))
                responses = [None, json.dumps(deployment), json.dumps(pods), json.dumps(GATE.BUILD["runtime"]), json.dumps(actual_artifact)]
                with patch.object(GATE, "ROOT", root), patch.object(GATE.SERVING, "check_target"), \
                     patch.object(GATE.SERVING, "check_restore"), patch.object(GATE.SERVING, "load_checkpoint", return_value={"pvc_identity": {"uid": "same"}}), \
                     patch.object(GATE.SERVING, "volume_identity", return_value=volumes), patch.object(GATE.SERVING, "write_private") as write, \
                     patch.object(GATE.SERVING, "run", side_effect=responses) as run, contextlib.redirect_stdout(io.StringIO()):
                    if succeeds:
                        GATE.main()
                        write.assert_called_once()
                    else:
                        with self.assertRaises(RuntimeError):
                            GATE.main()
                        write.assert_not_called()
                    for call in run.call_args_list:
                        self.assertIn("kind-energy-mlops-repro", call.args[0])
                        self.assertIn("energy-mlops-repro", call.args[0])


if __name__ == "__main__":
    unittest.main()
