"""Protect snapshot trust, overwrite prevention and destination-only DB writes."""

import io
import json
import os
import subprocess
import unittest
from pathlib import Path

import test_repro_snapshot as base

MODULE = base.MODULE


class EmptyRegistry:
    def search_registered_models(self, **kwargs):
        return []


class DestinationStorage(base.Storage):
    def __init__(self):
        super().__init__()
        self.objects = {}
        self.buckets = set()
        self.headers = {}

    def list_buckets(self):
        return {"Buckets": [{"Name": name} for name in self.buckets]}

    def create_bucket(self, Bucket):
        self.buckets.add(Bucket)

    def put_object(self, Bucket, Key, Body, ContentLength, IfNoneMatch, **headers):
        assert IfNoneMatch == "*" and Bucket in self.buckets
        if (Bucket, Key) in self.objects:
            raise RuntimeError("Object already exists")
        data = Body.read()
        assert len(data) == ContentLength
        self.objects[(Bucket, Key)] = data
        self.headers[(Bucket, Key)] = headers

    def get_object(self, Bucket, Key, IfMatch=None):
        return super().get_object(Bucket, Key, '"stable"')


class RestoreObjectTests(unittest.TestCase):
    def setUp(self):
        self.archive, _ = base.archive_bytes()
        with base.tarfile.open(fileobj=io.BytesIO(self.archive)) as archive:
            self.manifest = json.load(archive.extractfile("manifest.json"))

    def test_restoration_verifies_every_object_and_preserves_keys_and_headers(self):
        destination = DestinationStorage()
        result = MODULE.restore_objects(destination, EmptyRegistry(), self.manifest, io.BytesIO(self.archive))
        self.assertEqual(result, {"verified_objects": 2, "verified_bytes": 70014})
        self.assertEqual(destination.objects, base.Storage().objects)
        self.assertEqual(destination.headers[("energy-lake", "../../unsafe-key")]["Metadata"], {"purpose": "test"})

    def test_existing_registry_or_storage_prevents_all_writes(self):
        class PopulatedRegistry:
            def search_registered_models(self, **kwargs):
                return ["existing-model"]

        for populated_registry in (True, False):
            destination = DestinationStorage()
            if not populated_registry:
                destination.buckets.add("energy-lake")
                destination.objects[("energy-lake", "already-present")] = b"keep"
            previous = destination.objects.copy()
            with self.assertRaises(RuntimeError):
                MODULE.restore_objects(destination, PopulatedRegistry() if populated_registry else EmptyRegistry(),
                                       self.manifest, io.BytesIO(self.archive))
            self.assertEqual(destination.objects, previous)
            self.assertEqual(destination.headers, {})

    def test_corrupt_archive_is_not_uploaded_and_bad_destination_hash_is_detected(self):
        destination = DestinationStorage()
        corrupt = self.archive.replace(b"datasetdataset", b"Xatasetdataset", 1)
        with self.assertRaises(RuntimeError):
            MODULE.restore_objects(destination, EmptyRegistry(), self.manifest, io.BytesIO(corrupt))
        self.assertEqual(destination.objects, {})
        MODULE.restore_objects(destination, EmptyRegistry(), self.manifest, io.BytesIO(self.archive))
        destination.objects[("energy-lake", "../../unsafe-key")] = b"X" * 70000
        with self.assertRaises(RuntimeError):
            MODULE.check_objects(destination, self.manifest)


class RestoreWrapperTests(unittest.TestCase):
    def setUp(self):
        self.fixture = base.BackupWrapperTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        base.shutil.copy(base.ROOT / "scripts/restore-repro.sh", self.root / "scripts/restore-repro.sh")
        self.snapshot = self.root / ".repro/snapshots/snapshot-test"
        self.snapshot.mkdir(parents=True, mode=0o700)
        (self.snapshot / "postgres.dump").write_bytes(b"PGDMPfake-unit-test-dump")
        (self.snapshot / "objects.tar").write_bytes((self.root / "fixture.tar").read_bytes())
        MODULE.verify(self.snapshot)
        (self.root / ".repro/latest-snapshot").write_text(str(self.snapshot) + "\n")
        (self.root / "bin/kubectl").write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
root = Path(os.environ["FAKE_ROOT"])
args = sys.argv[1:]
with (root / "calls.jsonl").open("a") as log:
    log.write(json.dumps(args) + "\\n")
if "nodes" in args:
    print("energy-mlops-control-plane" if os.environ.get("BAD_NODE") else "energy-mlops-repro-control-plane", end="")
elif "energy-api" in args:
    pass
elif "--role" in args:
    print(json.dumps({"role":args[args.index("--role")+1]}))
elif "restore" in args:
    manifest = json.loads(sys.stdin.buffer.readline())
    assert sys.stdin.buffer.read() == (root / "fixture.tar").read_bytes()
    print(json.dumps({"verified_objects":2,"verified_bytes":70014}))
elif any("psql" in arg for arg in args):
    print(os.environ.get("DB_ROWS", "0"))
elif any("pg_restore" in arg for arg in args):
    sys.stdin.buffer.read()
    if os.environ.get("DB_FAIL"):
        raise SystemExit(3)
    (root / "db-restored").touch()
elif "check" in args:
    sys.stdin.read()
    model = json.loads((root / "plan.json").read_text())["model"]
    print(json.dumps({"model":model,"verified_objects":2,"verified_bytes":70014}))
elif "app=mlflow" in args:
    pass
elif "get" in args:
    print("fake workload inventory")
elif "scale" in args or "rollout" in args:
    pass
else:
    raise SystemExit("Unexpected kubectl operation")
''')

    def tearDown(self):
        self.fixture.tearDown()

    def run_restore(self, **extra):
        return subprocess.run(["bash", str(self.root / "scripts/restore-repro.sh")],
                              env={**self.fixture.env, **extra}, capture_output=True, text=True, timeout=30)

    def calls(self):
        path = self.root / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_success_restores_and_scales_only_the_rehearsal(self):
        result = self.run_restore()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "db-restored").exists())
        self.assertTrue((self.root / ".repro/restored-data.json").exists())
        scales = []
        for args in self.calls():
            context = args[args.index("--context") + 1]
            if "scale" in args or "restore" in args or any("pg_restore" in arg for arg in args):
                self.assertEqual(context, "kind-energy-mlops-repro")
                self.assertEqual(args[args.index("--namespace") + 1], "energy-mlops-repro")
                self.assertIn("--kubeconfig", args)
            if "scale" in args:
                scales.append(args[-1])
            if any("pg_restore" in arg for arg in args):
                command = next(arg for arg in args if "pg_restore" in arg)
                for flag in ("--single-transaction", "--clean", "--if-exists", "--exit-on-error"):
                    self.assertIn(flag, command)
        self.assertEqual(scales, ["--replicas=0", "--replicas=1"])
        self.assertNotIn("delete", [arg for args in self.calls() for arg in args])

    def test_wrong_node_and_existing_tracking_rows_block_mutations(self):
        for extra in ({"BAD_NODE":"1"}, {"DB_ROWS":"1"}):
            with self.subTest(extra=extra):
                result = self.run_restore(**extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / "db-restored").exists())
                self.assertFalse(any("restore" in args or "scale" in args for args in self.calls()))

    def test_changed_snapshot_is_rejected_without_rewriting_its_summary(self):
        original = (self.snapshot / "summary.json").read_bytes()
        (self.snapshot / "postgres.dump").write_bytes(b"PGDMPtampered-data")
        result = self.run_restore()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.snapshot / "summary.json").read_bytes(), original)
        self.assertFalse(any("restore" in args or "scale" in args for args in self.calls()))

    def test_database_error_leaves_mlflow_suspended_without_cleanup(self):
        result = self.run_restore(DB_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / ".repro/restored-data.json").exists())
        scales = [args[-1] for args in self.calls() if "scale" in args]
        self.assertEqual(scales, ["--replicas=0"])
        self.assertIn("fase database", result.stderr)


if __name__ == "__main__":
    unittest.main()
