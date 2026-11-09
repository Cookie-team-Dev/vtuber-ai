import unittest
from unittest import mock

import controller


class CliExitTests(unittest.TestCase):
    def test_slash_quit_stops_without_sending_a_chat_turn(self):
        controller._AUTONOMOUS_STOP.clear()
        with (
            mock.patch.object(controller, "start_runtime"),
            mock.patch.object(controller, "schedule_idle_check"),
            mock.patch.object(controller, "stop_runtime") as stop_runtime,
            mock.patch.object(controller, "ask_ai") as ask_ai,
            mock.patch("builtins.input", return_value="  /QuIt  "),
        ):
            controller.run_cli()

        ask_ai.assert_not_called()
        stop_runtime.assert_called_once_with()

    def test_resource_command_is_handled_locally_without_chat_generation(self):
        controller._AUTONOMOUS_STOP.clear()
        with (
            mock.patch.object(controller, "start_runtime"),
            mock.patch.object(controller, "schedule_idle_check"),
            mock.patch.object(controller, "stop_runtime"),
            mock.patch.object(controller, "handle_resource_command", return_value="RAM report") as command,
            mock.patch.object(controller, "ask_ai") as ask_ai,
            mock.patch("builtins.input", side_effect=["/resources", "/quit"]),
        ):
            controller.run_cli()

        command.assert_called_once_with("/resources")
        ask_ai.assert_not_called()


if __name__ == "__main__":
    unittest.main()
