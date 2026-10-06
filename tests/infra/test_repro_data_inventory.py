"""Protect the restore preflight using clients that expose only read APIs."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("repro_inventory", ROOT / "scripts/inventory-repro-data.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Page(list):
    def __init__(self, items, token=None):
        super().__init__(items)
        self.token = token


class Registry:
    def __init__(self, populated=True, version=17, run_id=MODULE.EXPECTED_RUN_ID):
        self.populated = populated
        self.model = SimpleNamespace(version=str(version), run_id=run_id, source="models:/m-example")

    def search_registered_models(self, **kwargs):
        return Page([SimpleNamespace(name=MODULE.MODEL_NAME)] if self.populated else [])

    def get_model_version_by_alias(self, name, alias):
        return self.model

    def get_run(self, run_id):
        return SimpleNamespace(info=SimpleNamespace(artifact_uri="s3://mlflow-artifacts/example/artifacts"))

    def get_model_version_download_uri(self, name, version):
        return "s3://mlflow-artifacts/example/model"


class Storage:
    def __init__(self, objects=None):
        self.objects = objects or {}

    def list_buckets(self):
        return {"Buckets": [{"Name": name} for name in self.objects]}

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, Bucket):
        yield {"Contents": [{"Size": size} for size in self.objects[Bucket]]}
        yield {}  # Empty pages must not inflate the inventory.


class InventoryTests(unittest.TestCase):
    def test_source_reports_identity_and_bucket_sizes_without_mutation(self):
        result = MODULE.inventory("source", Registry(), Storage({"energy-lake": [3, 5], "mlflow-artifacts": [7]}))
        self.assertEqual(result["model"]["version"], 17)
        self.assertEqual(result["model"]["run_id"], MODULE.EXPECTED_RUN_ID)
        self.assertEqual(result["buckets"]["energy-lake"], {"present": True, "objects": 2, "bytes": 8})

    def test_champion_change_or_missing_bucket_stops_preflight(self):
        for registry, storage in (
            (Registry(version=18), Storage()),
            (Registry(run_id="different-run"), Storage()),
            (Registry(), Storage({"energy-lake": []})),
        ):
            with self.subTest(registry=registry.model):
                with self.assertRaises(RuntimeError):
                    MODULE.inventory("source", registry, storage)

    def test_empty_destination_can_have_absent_or_empty_buckets(self):
        for storage in (Storage(), Storage({"energy-lake": [], "mlflow-artifacts": []})):
            result = MODULE.inventory("destination", Registry(populated=False), storage)
            self.assertEqual(result["registered_models"], [])
            self.assertIsNone(result["model"])

    def test_existing_destination_data_stops_preflight(self):
        for registry, storage in (
            (Registry(), Storage()),
            (Registry(populated=False), Storage({"mlflow-artifacts": [5]})),
            (Registry(populated=False), Storage({"unexpected": []})),
        ):
            with self.assertRaises(RuntimeError):
                MODULE.inventory("destination", registry, storage)

    def test_registry_inventory_follows_pagination(self):
        class PaginatedRegistry(Registry):
            def search_registered_models(self, page_token=None, **kwargs):
                if page_token is None:
                    return Page([SimpleNamespace(name="first")], token="next")
                assert page_token == "next"
                return Page([SimpleNamespace(name=MODULE.MODEL_NAME)])

        result = MODULE.inventory("source", PaginatedRegistry(), Storage({bucket: [] for bucket in MODULE.BUCKETS}))
        self.assertEqual(len(result["registered_models"]), 2)


if __name__ == "__main__":
    unittest.main()
