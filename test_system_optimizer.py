import unittest
import contextlib
import io
from pathlib import Path
from unittest import mock

import system_optimizer
import controller
import kookie_ui


class FakeProcess:
    def __init__(self):
        self.pid = 123
        self._affinity = [0, 1, 2, 3]

    def create_time(self):
        return 1.0

    def name(self):
        return "python"

    def cpu_affinity(self, value=None):
        if value is not None:
            self._affinity = list(value)
        return list(self._affinity)


class SystemOptimizerTests(unittest.TestCase):
    def tearDown(self):
        system_optimizer._CPU_CHANGES.clear()

    def test_cpu_saver_is_limited_and_restorable(self):
        process = FakeProcess()
        with mock.patch.object(system_optimizer, "_target_processes", return_value=[process]):
            result = system_optimizer.apply_cpu_saver()
            self.assertEqual(process.cpu_affinity(), [0, 1])
            self.assertIn("2/4 CPUs", result)

            restored = system_optimizer.restore_cpu_settings()

        self.assertEqual(process.cpu_affinity(), [0, 1, 2, 3])
        self.assertIn("restored", restored)

    def test_failed_cpu_restore_keeps_original_settings_for_retry(self):
        process = FakeProcess()
        with mock.patch.object(system_optimizer, "_target_processes", return_value=[process]):
            system_optimizer.apply_cpu_saver()

        key = (process.pid, process.create_time())
        with mock.patch.object(
            process,
            "cpu_affinity",
            side_effect=system_optimizer.psutil.AccessDenied(pid=process.pid),
        ):
            result = system_optimizer.restore_cpu_settings()

        self.assertIn("could not restore", result)
        self.assertIn(key, system_optimizer._CPU_CHANGES)

        result = system_optimizer.restore_cpu_settings()

        self.assertEqual(process.cpu_affinity(), [0, 1, 2, 3])
        self.assertIn("restored", result)
        self.assertNotIn(key, system_optimizer._CPU_CHANGES)

    def test_memory_unload_requires_a_loopback_ollama_address(self):
        with self.assertRaises(ValueError):
            system_optimizer.unload_model("qwen3:4b", "http://example.com:11434")

    def test_cpu_target_discovery_includes_only_amika_tts_from_this_checkout(self):
        engine = Path(__file__).resolve().parent / "openamika" / "vendor" / "tts-engine"
        amika = mock.Mock()
        amika.info = {"cmdline": ["python", "main.py"], "cwd": str(engine), "uids": None}
        unrelated = mock.Mock()
        unrelated.info = {"cmdline": ["python", "main.py"], "cwd": "/tmp", "uids": None}
        with mock.patch.object(system_optimizer.psutil, "process_iter", return_value=[amika, unrelated]):
            self.assertEqual(list(system_optimizer._amika_tts_processes()), [amika])

    def test_unload_verifies_that_ollama_released_the_model(self):
        loaded = [{"name": "qwen3:4b", "size": 2 * 1024 ** 3}]
        with (
            mock.patch.object(system_optimizer, "_loaded_models", side_effect=[loaded, []]),
            mock.patch.object(system_optimizer.requests, "post") as post,
        ):
            post.return_value.raise_for_status.return_value = None
            result = system_optimizer.unload_model("qwen3:4b", "http://127.0.0.1:11434")

        post.assert_called_once_with(
            "http://127.0.0.1:11434/api/generate",
            json={"model": "qwen3:4b", "keep_alive": 0},
            timeout=30,
        )
        self.assertIn("2.0 GiB RAM", result)
        self.assertIn("load it again", result)

    def test_controller_requires_a_specific_memory_unload_command(self):
        with mock.patch.object(controller.system_optimizer, "unload_model", return_value="unloaded") as unload:
            prompt = controller.handle_resource_command("/optimize memory")
            unload.assert_not_called()
            self.assertIn("/optimize memory unload", prompt)

            result = controller.handle_resource_command("/optimize memory unload")

        unload.assert_called_once_with(controller.MODEL, controller._OLLAMA_BASE)
        self.assertEqual(result, "unloaded")

    def test_controller_routes_cpu_limit_and_restore_commands(self):
        with (
            mock.patch.object(controller.system_optimizer, "apply_cpu_saver", return_value="limited") as limit,
            mock.patch.object(controller.system_optimizer, "restore_cpu_settings", return_value="restored") as restore,
        ):
            self.assertEqual(controller.handle_resource_command("/optimize cpu"), "limited")
            self.assertEqual(controller.handle_resource_command("/optimize restore"), "restored")

        limit.assert_called_once_with()
        restore.assert_called_once_with()

    def test_window_routes_resource_slash_commands_to_local_controls(self):
        window = object.__new__(kookie_ui.KookieWindow)
        window.system = mock.Mock()
        with mock.patch.object(controller, "handle_resource_command", return_value="CPU report\nRAM report") as handle:
            kookie_ui.KookieWindow.slash(window, "/resources")

        handle.assert_called_once_with("/resources")
        self.assertEqual([call.args[0] for call in window.system.call_args_list], ["CPU report", "RAM report"])

    def test_runtime_shutdown_restores_cpu_affinity(self):
        with (
            mock.patch.multiple(controller, animation_thread=None, idle_timer=None, running=True, vts=None),
            mock.patch.object(controller, "clear_expression"),
            mock.patch.object(controller, "save_learning_data"),
            mock.patch.object(controller, "shutdown_browser"),
            mock.patch.object(controller.system_optimizer, "restore_cpu_settings", return_value="restored") as restore,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            controller.stop_runtime()

        restore.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
