"""The launcher sources .env, so a real ./Kookie run has its cloud keys.

fastai reads .env itself, but only when the key is already in the environment.
The launcher used to export only OLLAMA_BASE, so every reply fell through to the
slow local model even though the keys were sitting in .env one directory up.
These tests check the sourcing logic in isolation, without spawning a child.
"""

import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(ROOT, "Kookie")


class LauncherEnvTests(unittest.TestCase):
    def test_launcher_is_bash_and_runs_help(self):
        # bash -n is the syntax check; --help is the smoke test that the
        # argument parser and the env sourcing both survive to startup.
        syntax = subprocess.run(
            ["bash", "-n", LAUNCHER], capture_output=True, text=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        help_run = subprocess.run(
            [LAUNCHER, "--help"], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn("--help", help_run.stdout)
        self.assertIn("--discord", help_run.stdout)

    def test_dotenv_is_sourced_into_a_clean_environment(self):
        # A fresh shell with no cloud keys exported must still pick them up
        # from .env, because that is the whole point of the sourcing change.
        # The sourcing block sits right after the DIR resolution, so sourcing
        # the launcher up to that point and checking the variables is enough -
        # it does not need the rest of the startup path. The range must run
        # past the _KOOK_ENV= assignment and through the unset that follows,
        # because stopping at the assignment means the .env is never sourced.
        script = (
            "export -n GEMINI_API_KEY OPENROUTER_API_KEY GROQ_API_KEY "
            "KOOKIE_OMNIROUTE_API_KEY KOOKIE_TTS_SPEED; "
            # It is sourced into the current shell (not piped to a child bash), because a
            # piped bash is a subshell and its exports would not survive back
            # here for the assertions to see. The fragment is written beside the
            # real launcher (not to /tmp) so the script's own BASH_SOURCE
            # resolves DIR to the project directory.
            "sed -n '1,/^unset _KOOK_ENV/p' ./Kookie > ./_kook_src.sh; "
            ". ./_kook_src.sh; "
            "rm -f ./_kook_src.sh; "
            "test \"$DIR\" = \"$(pwd)\" || { echo \"DIR=$DIR\"; exit 1; }; "
            "test -n \"$GEMINI_API_KEY\" && echo has-gemini; "
            "test -n \"$OPENROUTER_API_KEY\" && echo has-openrouter; "
            "test -n \"$KOOKIE_TTS_SPEED\" && echo has-tts-speed"
        )
        run = subprocess.run(
            ["bash", "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertIn("has-gemini", run.stdout, run.stdout + run.stderr)
        self.assertIn("has-openrouter", run.stdout, run.stdout + run.stderr)
        self.assertIn("has-tts-speed", run.stdout, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()