import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import second_brain


class SecondBrainTests(unittest.TestCase):
    def setUp(self):
        self._original_file = second_brain._SECOND_BRAIN_FILE
        self._original_state = {
            "thoughts": list(second_brain._SECOND_BRAIN_STATE.get("thoughts", [])),
            "problems": dict(second_brain._SECOND_BRAIN_STATE.get("problems", {})),
            "insights": list(second_brain._SECOND_BRAIN_STATE.get("insights", [])),
            "background_queue": list(second_brain._SECOND_BRAIN_STATE.get("background_queue", [])),
        }
        self._tmp = tempfile.TemporaryDirectory()
        second_brain._SECOND_BRAIN_FILE = Path(self._tmp.name) / "second_brain_memory.json"
        second_brain._SECOND_BRAIN_STATE["thoughts"] = []
        second_brain._SECOND_BRAIN_STATE["problems"] = {}
        second_brain._SECOND_BRAIN_STATE["insights"] = []
        second_brain._SECOND_BRAIN_STATE["background_queue"] = []
        second_brain._NEXT_THOUGHT_ID = 1
        second_brain._NEXT_PROBLEM_ID = 1

    def tearDown(self):
        second_brain._SECOND_BRAIN_FILE = self._original_file
        second_brain._SECOND_BRAIN_STATE["thoughts"] = self._original_state["thoughts"]
        second_brain._SECOND_BRAIN_STATE["problems"] = self._original_state["problems"]
        second_brain._SECOND_BRAIN_STATE["insights"] = self._original_state["insights"]
        second_brain._SECOND_BRAIN_STATE["background_queue"] = self._original_state["background_queue"]
        self._tmp.cleanup()

    def test_think_deeply_stores_and_persists(self):
        result = second_brain.think_deeply("cats", "They purr when happy.", confidence=0.9)
        self.assertIn("#1", result)
        self.assertEqual(len(second_brain._SECOND_BRAIN_STATE["thoughts"]), 1)
        self.assertTrue(second_brain._SECOND_BRAIN_FILE.exists())
        data = json.loads(second_brain._SECOND_BRAIN_FILE.read_text(encoding="utf-8"))
        self.assertEqual(data["thoughts"][0]["topic"], "cats")
        self.assertEqual(data["thoughts"][0]["confidence"], 0.9)

    def test_save_failure_does_not_lose_the_in_memory_state(self):
        # think_deeply() already wrote the file once; now force the next save to
        # fail and check the in-memory state survives.
        second_brain.think_deeply("topic", "content")
        with mock.patch.object(second_brain.os, "replace", side_effect=OSError("read-only")):
            self.assertFalse(second_brain._save_state())
        self.assertEqual(len(second_brain._SECOND_BRAIN_STATE["thoughts"]), 1)

    def test_start_and_complete_problem(self):
        second_brain.start_problem("build", "Ship the feature.")
        self.assertEqual(len(second_brain._SECOND_BRAIN_STATE["problems"]), 1)
        second_brain.solve_problem_step(1, "Write tests", "passing")
        second_brain.complete_problem(1, "Done.")
        problem = second_brain._SECOND_BRAIN_STATE["problems"]["1"]
        self.assertEqual(problem["status"], "completed")
        self.assertEqual(problem["result"], "Done.")
        self.assertGreaterEqual(len(problem["steps"]), 1)

    def test_background_queue_round_trip(self):
        second_brain.queue_background_thought("dream", "Dream of flying cats.")
        self.assertEqual(len(second_brain._SECOND_BRAIN_STATE["background_queue"]), 1)
        second_brain.process_background_queue()
        self.assertEqual(second_brain._SECOND_BRAIN_STATE["background_queue"], [])
        self.assertGreaterEqual(len(second_brain._SECOND_BRAIN_STATE["thoughts"]), 1)


if __name__ == "__main__":
    unittest.main()