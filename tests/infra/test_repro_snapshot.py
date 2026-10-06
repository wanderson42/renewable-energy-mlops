"""Exercise streamed export, integrity verification and backup failure boundaries."""

import importlib.util
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("repro_snapshot", ROOT / "scripts/snapshot-repro-objects.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MODEL_KEY = "6/models/m-example/artifacts/MLmodel"


class Registry:
    def get_model_version_by_alias(self, name, alias):
        return SimpleNamespace(version="17", run_id=MODULE.RUN_ID, source="models:/m-example")

    def get_model_version_download_uri(self, name, version):
        return "s3://mlflow-artifacts/6/models/m-example/artifacts"


class Storage:
    def __init__(self):
        self.objects = {("energy-lake", "../../unsafe-key"): b"dataset" * 10000,
                        ("mlflow-artifacts", MODEL_KEY): b"model metadata"}
        self.changed = False

    def list_buckets(self):
        return {"Buckets": [{"Name": name} for name in MODULE.BUCKETS]}

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, Bucket):
        yield {"Contents": [{"Key": key, "Size": len(data), "ETag": '"changed"' if self.changed else '"stable"',
                             "LastModified": datetime(2026, 10, 5, tzinfo=timezone.utc)}
                            for (bucket, key), data in self.objects.items() if bucket == Bucket]}

    def get_object(self, Bucket, Key, IfMatch):
        assert IfMatch == '"stable"'
        data = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(data), "ContentLength": len(data), "ETag": '"stable"',
                "ContentType": "application/octet-stream", "Metadata": {"purpose": "test"}}


def archive_bytes(storage=None):
    storage = storage or Storage()
    plan = MODULE.inventory(storage, Registry())
    output = io.BytesIO()
    MODULE.export(storage, Registry(), plan, output)
    return output.getvalue(), plan


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        (self.directory / "postgres.dump").write_bytes(b"PGDMPfake-unit-test-dump")

    def tearDown(self):
        self.temp.cleanup()

    def test_roundtrip_preserves_bytes_and_keys_without_path_extraction(self):
        archive, plan = archive_bytes()
        (self.directory / "objects.tar").write_bytes(archive)
        result = MODULE.verify(self.directory)
        self.assertEqual(result["buckets"]["energy-lake"], {"objects": 1, "bytes": 70000})
        self.assertEqual(result["model"]["run_id"], MODULE.RUN_ID)
        self.assertEqual(result["files"]["objects.tar"]["sha256"], MODULE.file_hash(self.directory / "objects.tar"))
        self.assertTrue((self.directory / "SHA256SUMS").exists())
        self.assertEqual(plan["objects"][0]["key"], "../../unsafe-key")
        self.assertFalse((self.directory.parent / "unsafe-key").exists())

    def test_corrupt_object_and_missing_footer_are_rejected(self):
        original, _ = archive_bytes()
        for remove_footer, corrupt in ((False, True), (True, False)):
            with self.subTest(remove_footer=remove_footer):
                out = io.BytesIO()
                with tarfile.open(fileobj=io.BytesIO(original)) as source, tarfile.open(fileobj=out, mode="w") as target:
                    for item in source:
                        if remove_footer and item.name == "manifest.json":
                            continue
                        data = source.extractfile(item).read()
                        if corrupt and item.name == "objects/00000000":
                            data = b"X" + data[1:]
                        target.addfile(item, io.BytesIO(data))
                (self.directory / "objects.tar").write_bytes(out.getvalue())
                with self.assertRaises(RuntimeError):
                    MODULE.verify(self.directory)
                self.assertFalse((self.directory / "summary.json").exists())

    def test_mutation_during_export_prevents_manifest_completion(self):
        class ChangingStorage(Storage):
            def get_object(self, **kwargs):
                result = super().get_object(**kwargs)
                self.changed = True
                return result

        storage = ChangingStorage()
        plan = MODULE.inventory(storage, Registry())
        output = io.BytesIO()
        with self.assertRaises(RuntimeError):
            MODULE.export(storage, Registry(), plan, output)
        (self.directory / "objects.tar").write_bytes(output.getvalue())
        with self.assertRaises(RuntimeError):
            MODULE.verify(self.directory)

    def test_stale_inventory_and_missing_model_artifact_are_rejected(self):
        storage = Storage()
        plan = MODULE.inventory(storage, Registry())
        storage.changed = True
        with self.assertRaises(RuntimeError):
            MODULE.export(storage, Registry(), plan, io.BytesIO())
        storage = Storage()
        del storage.objects[("mlflow-artifacts", MODEL_KEY)]
        with self.assertRaises(RuntimeError):
            MODULE.inventory(storage, Registry())

    def test_unexpected_archive_paths_are_rejected(self):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            item = tarfile.TarInfo("../../unsafe-key")
            item.size = 1
            archive.addfile(item, io.BytesIO(b"x"))
        (self.directory / "objects.tar").write_bytes(output.getvalue())
        with self.assertRaises(RuntimeError):
            MODULE.verify(self.directory)


@unittest.skipUnless(shutil.which("bash"), "Bash is required")
class BackupWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / ".repro").mkdir(mode=0o700)
        (self.root / ".repro/kubeconfig").write_text("fake")
        (self.root / "bin").mkdir()
        for name in ("backup-repro.sh", "snapshot-repro-objects.py", "inventory-repro-data.sh", "inventory-repro-data.py"):
            shutil.copy(ROOT / "scripts" / name, self.root / "scripts" / name)
        archive, plan = archive_bytes()
        (self.root / "fixture.tar").write_bytes(archive)
        (self.root / "plan.json").write_text(json.dumps(plan))
        (self.root / "bin/kubectl").write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
root = Path(os.environ["FAKE_ROOT"])
args = sys.argv[1:]
with (root / "calls.jsonl").open("a") as log:
    log.write(json.dumps(args) + "\\n")
if "--role" in args:
    role = args[args.index("--role") + 1]
    print(json.dumps({"role": role}))
elif "inventory" in args:
    print((root / "plan.json").read_text())
elif "export" in args:
    if os.environ.get("FAIL_EXPORT"):
        raise SystemExit(7)
    sys.stdout.buffer.write((root / "fixture.tar").read_bytes())
elif any("pg_dump" in arg for arg in args):
    sys.stdout.buffer.write(b"PGDMPfake-unit-test-dump")
elif "pg_restore" in args:
    assert "--file=/dev/null" in args and not any(arg.startswith("--dbname") for arg in args)
    sys.stdin.buffer.read()
elif "get" in args:
    print("fake pod inventory")
else:
    raise SystemExit("Unexpected kubectl operation")
''')
        (self.root / "bin/curl").write_text('''#!/usr/bin/env python3
import json
print(json.dumps({"model_name":"ensemble_lgb_xgb_rf_bahia","version":17,"run_id":"1d13a61244c54f06aa70f43a9993ea37"}))
''')
        for name in ("kubectl", "curl"):
            (self.root / "bin" / name).chmod(0o755)
        self.env = {**os.environ, "PATH": str(self.root / "bin") + os.pathsep + os.environ["PATH"], "FAKE_ROOT": str(self.root)}

    def tearDown(self):
        self.temp.cleanup()

    def run_backup(self, **extra):
        return subprocess.run(["bash", str(self.root / "scripts/backup-repro.sh")], env={**self.env, **extra},
                              capture_output=True, text=True, timeout=30)

    def test_success_uses_explicit_contexts_and_protects_snapshot_files(self):
        result = self.run_backup()
        self.assertEqual(result.returncode, 0, result.stderr)
        snapshot = Path((self.root / ".repro/latest-snapshot").read_text().strip())
        self.assertTrue((snapshot / "summary.json").exists())
        self.assertFalse((snapshot / "INCOMPLETE").exists())
        self.assertEqual(snapshot.stat().st_mode & 0o777, 0o700)
        self.assertEqual((snapshot / "postgres.dump").stat().st_mode & 0o777, 0o600)
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        for args in calls:
            context = args[args.index("--context") + 1]
            self.assertIn(context, ("kind-energy-mlops", "kind-energy-mlops-repro"))
            if context == "kind-energy-mlops-repro":
                self.assertIn("--kubeconfig", args)
                self.assertTrue("destination" in args or "get" in args)
            if "export" in args or "pg_restore" in args or any("pg_dump" in arg for arg in args):
                self.assertEqual(context, "kind-energy-mlops")
            self.assertNotIn("apply", args)
            self.assertNotIn("delete", args)
            self.assertNotIn("scale", args)

    def test_failed_backup_preserves_partial_files_and_previous_good_pointer(self):
        pointer = self.root / ".repro/latest-snapshot"
        pointer.write_text("previous-good-snapshot\n")
        result = self.run_backup(FAIL_EXPORT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(pointer.read_text(), "previous-good-snapshot\n")
        snapshots = list((self.root / ".repro/snapshots").iterdir())
        self.assertEqual(len(snapshots), 1)
        self.assertTrue((snapshots[0] / "INCOMPLETE").exists())
        self.assertFalse((snapshots[0] / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
