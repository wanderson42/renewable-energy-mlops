"""Verify client isolation and process cleanup without Poetry or Kubernetes."""

import contextlib
import importlib.util
import io
import json
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("repro_client", ROOT / "scripts/repro-client.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(MODULE, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def test_child_settings_override_source_without_changing_parent(self):
        original = {"PREFECT_API_URL": "http://localhost:4200/api", "PREFECT_API_KEY": "source-secret",
                    "PREFECT_HOME": "/source", "PREFECT_UNKNOWN_SETTING": "source",
                    "AWS_PROFILE": "source-cloud", "AWS_SESSION_TOKEN": "source-token",
                    "API_URL": "http://localhost:8000", "RUSTFS_ROOT_PASSWORD": "source-password"}
        with patch.dict(os.environ, original, clear=True):
            env = MODULE.child_environment("repro-user", "repro-secret")
            self.assertEqual(dict(os.environ), original)
        self.assertEqual(env["API_URL"], "http://127.0.0.1:18000")
        self.assertEqual(env["MLFLOW_TRACKING_URI"], "http://127.0.0.1:15000")
        self.assertEqual(env["RUSTFS_ENDPOINT"], "http://127.0.0.1:19000")
        self.assertEqual(env["PREFECT_API_URL"], "http://127.0.0.1:14200/api")
        for key in ("PREFECT_API_KEY", "PREFECT_API_AUTH_STRING", "PREFECT_SERVER_API_AUTH_STRING"):
            self.assertNotIn(key, env)
        self.assertNotIn("PREFECT_UNKNOWN_SETTING", env)
        self.assertNotIn("AWS_PROFILE", env)
        self.assertNotIn("AWS_SESSION_TOKEN", env)
        self.assertEqual(env["PREFECT_PROFILE"], "repro")
        self.assertEqual(env["PREFECT_HOME"], str(self.root / ".repro/client/prefect"))
        self.assertEqual(env["PREFECT_SERVER_DATABASE_CONNECTION_URL"],
                         "sqlite+aiosqlite:///" + str(self.root / ".repro/client/prefect/orchestration.db"))
        self.assertEqual(env["RUSTFS_ROOT_PASSWORD"], "repro-secret")
        self.assertFalse((self.root / ".env").exists())

    def test_commands_bind_loopback_and_use_only_explicit_rehearsal_target(self):
        commands = MODULE.commands()
        for name in ("api", "mlflow", "rustfs"):
            command = commands[name]
            self.assertEqual(command[command.index("--context") + 1], "kind-energy-mlops-repro")
            self.assertEqual(command[command.index("--namespace") + 1], "energy-mlops-repro")
            self.assertTrue(command[command.index("--kubeconfig") + 1].endswith(".repro/kubeconfig"))
            self.assertEqual(command[command.index("--address") + 1], "127.0.0.1")
        self.assertIn("14200", commands["prefect"])
        self.assertIn("--server.port=18501", commands["dashboard"])
        self.assertFalse(any("config" in c or "pkill" in c for c in commands.values()))

    def test_busy_port_closes_probe_sockets_and_does_not_kill_processes(self):
        listeners = [Mock(), Mock()]
        listeners[1].bind.side_effect = OSError("occupied")
        with patch.object(MODULE.socket, "socket", side_effect=listeners), \
             patch.object(MODULE.os, "killpg") as kill, self.assertRaisesRegex(RuntimeError, "ocupada"):
            MODULE.require_free_ports()
        for listener in listeners:
            listener.close.assert_called_once()
        kill.assert_not_called()

    def test_wrong_node_blocks_every_child_and_local_state_creation(self):
        with patch.object(MODULE.shutil, "which", return_value="fake-cli"), \
             patch.object(MODULE.SERVING, "run", return_value="energy-mlops-control-plane"), \
             patch.object(MODULE.subprocess, "Popen") as spawn, self.assertRaisesRegex(RuntimeError, "Node"):
            MODULE.start()
        spawn.assert_not_called()
        self.assertFalse((self.root / ".repro/client").exists())

    def test_cleanup_signals_only_owned_groups_and_escalates_stalled_child(self):
        first, second = Mock(pid=123), Mock(pid=456)
        first.wait.side_effect = [subprocess.TimeoutExpired("child", 8), 0]
        with patch.object(MODULE.os, "killpg") as kill:
            MODULE.stop_children([("first", first), ("second", second)])
        self.assertEqual(kill.call_args_list, [unittest.mock.call(456, signal.SIGTERM),
                                             unittest.mock.call(123, signal.SIGTERM),
                                             unittest.mock.call(123, signal.SIGKILL)])

    def test_failed_start_cleans_owned_processes_and_preserves_existing_profile(self):
        home = self.root / ".repro/client/prefect"
        home.mkdir(parents=True)
        profile = home / "profiles.toml"
        profile.write_text('active = "repro"\n[profiles.repro]\n')
        previous = profile.read_bytes()
        process = Mock(pid=123)
        handler = signal.getsignal(signal.SIGTERM)
        with patch.object(MODULE.shutil, "which", return_value="fake-cli"), \
             patch.object(MODULE, "environment_from_target", return_value=MODULE.child_environment("user", "secret")), \
             patch.object(MODULE, "require_free_ports"), \
             patch.object(MODULE, "commands", return_value={"prefect": ["fake-cli"]}), \
             patch.object(MODULE.subprocess, "Popen", return_value=process) as spawn, \
             patch.object(MODULE, "wait_services", side_effect=RuntimeError("failed health")), \
             patch.object(MODULE, "stop_children") as stop, contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError, "failed health"):
            MODULE.start()
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        stop.assert_called_once_with([("prefect", process)])
        self.assertEqual(profile.read_bytes(), previous)
        self.assertEqual(signal.getsignal(signal.SIGTERM), handler)

    def test_failed_probe_keeps_raw_output_private_and_previous_receipt_unchanged(self):
        folder = self.root / ".repro/client"
        folder.mkdir(parents=True)
        receipt = self.root / ".repro/client-validation.json"
        receipt.write_text('{"previous": true}')
        result = subprocess.CompletedProcess([], 1, "", "secret-in-failure")
        output = io.StringIO()
        with patch.object(MODULE.subprocess, "run", return_value=result), \
             contextlib.redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "Gate do cliente") as error:
            MODULE.run_probe({})
        self.assertNotIn("secret-in-failure", str(error.exception) + output.getvalue())
        self.assertEqual(json.loads(receipt.read_text()), {"previous": True})
        self.assertEqual((folder / "probe.log").stat().st_mode & 0o777, 0o600)

    def test_early_child_exit_interrupts_gate(self):
        with self.assertRaisesRegex(RuntimeError, "Processo dashboard terminou"):
            MODULE.require_children_alive([("dashboard", Mock(poll=Mock(return_value=1)))])

    def test_effective_empty_auth_is_rejected_even_when_health_could_pass(self):
        api_url = MODULE.urls()["prefect"] + "/api"
        database = "sqlite+aiosqlite:///" + str(self.root / ".repro/client/prefect/orchestration.db")
        settings = SimpleNamespace(home=self.root / ".repro/client/prefect",
                                   api=SimpleNamespace(url=api_url, key=None, auth_string=None),
                                   server=SimpleNamespace(api=SimpleNamespace(auth_string=None),
                                       ui=SimpleNamespace(api_url=api_url),
                                       database=SimpleNamespace(connection_url=Mock(get_secret_value=lambda: database))))
        MODULE.check_prefect_settings(settings)
        for value in ("", "user:secret"):
            settings.server.api.auth_string = value
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "Autenticação"):
                MODULE.check_prefect_settings(settings)
        settings.server.api.auth_string = None
        settings.server.ui.api_url = "http://localhost:4200/api"
        with self.assertRaisesRegex(RuntimeError, "não está isolada"):
            MODULE.check_prefect_settings(settings)

    def test_ui_gate_checks_live_config_and_read_only_count_without_auth_header(self):
        target = MODULE.urls()["prefect"]
        with patch.object(MODULE, "fetch", side_effect=[{"api_url": target + "/api", "auth": None}, 0]) as fetch:
            result = MODULE.check_prefect_ui()
        self.assertEqual(result["flow_runs_count"], 0)
        self.assertEqual(fetch.call_args_list, [unittest.mock.call(target + "/ui-settings", True),
                                               unittest.mock.call(target + "/api/flow_runs/count", True, payload={})])
        self.assertEqual(result["browser_e2e"], "not exercised by this gate")

    def test_ui_gate_rejects_wrong_endpoint_auth_and_unauthorized_data_query(self):
        target = MODULE.urls()["prefect"]
        for settings in ({"api_url": "http://localhost:4200/api", "auth": None},
                         {"api_url": target + "/api", "auth": "BASIC"}):
            with patch.object(MODULE, "fetch", return_value=settings) as fetch, self.assertRaises(RuntimeError):
                MODULE.check_prefect_ui()
            self.assertEqual(fetch.call_count, 1)
        unauthorized = HTTPError(target + "/api/flow_runs/count", 401, "Unauthorized", {}, None)
        with patch.object(MODULE, "fetch", side_effect=[{"api_url": target + "/api", "auth": None}, unauthorized]), \
             self.assertRaises(HTTPError):
            MODULE.check_prefect_ui()

    def test_count_transport_posts_json_without_authorization(self):
        response = io.BytesIO(b"0")
        response.status = 200
        opener = Mock()
        opener.open.return_value = response
        target = MODULE.urls()["prefect"] + "/api/flow_runs/count"
        with patch.object(MODULE, "build_opener", return_value=opener):
            self.assertEqual(MODULE.fetch(target, True, payload={}), 0)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, target)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data), {})
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertIsNone(request.get_header("Authorization"))


if __name__ == "__main__":
    unittest.main()
