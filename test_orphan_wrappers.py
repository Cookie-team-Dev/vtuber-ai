"""Tests for the orphaned-module tool wrappers wired into controller.py."""
import controller


def test_lipsync_state_string():
    result = controller.lipsync_state()
    assert isinstance(result, str)
    assert "active:" in result


def test_osu_status_string():
    result = controller.osu_status()
    assert isinstance(result, str)
    assert "running:" in result


def test_content_filter_stats_string():
    result = controller.content_filter_stats()
    assert isinstance(result, str)
    assert "enabled:" in result


def test_amika_brain_retrieve_no_memory():
    result = controller.amika_brain_retrieve("nonexistent topic")
    assert isinstance(result, str)


def test_parse_speech_codes_empty():
    assert controller.parse_speech_codes("") == ""


def test_parse_speech_codes_with_codes():
    result = controller.parse_speech_codes("[happy] hello there")
    assert isinstance(result, str)


def test_filter_input_text_passthrough():
    assert controller.filter_input_text("plain text") == "plain text"


def test_monid_balance_error_string():
    result = controller.monid_balance()
    assert isinstance(result, str)
    assert "Error" in result or "monid" in result.lower()


def test_monid_whoami_error_string():
    result = controller.monid_whoami()
    assert isinstance(result, str)


def test_monid_discover_error_string():
    result = controller.monid_discover("test")
    assert isinstance(result, str)


def test_monid_runs_list_error_string():
    result = controller.monid_runs_list()
    assert isinstance(result, str)


def test_sanitize_speech_passthrough():
    assert controller.sanitize_speech("hello world") == "hello world"


def test_strip_controls_removes_control_chars():
    assert controller.strip_controls("a【想想】b") == "ab"


def test_eval_check_answer_contains():
    result = controller.eval_check_answer("contains", "q", "the number 42", "42")
    assert result.startswith("1.0:")


def test_eval_check_answer_words():
    result = controller.eval_check_answer("words", "q", "one two three", "3")
    assert result.startswith("1.0:")


def test_training_emotions_string():
    result = controller.training_emotions()
    assert isinstance(result, str)
    assert "neutral" in result.lower()


def test_load_rows_missing_file():
    result = controller.load_rows("/nonexistent/path/file.jsonl")
    assert isinstance(result, str)
    assert "Error" in result


def test_train_dry_run_missing_files():
    result = controller.train_dry_run("/nope/train.jsonl", "/nope/eval.jsonl", "kookie-3b")
    assert isinstance(result, str)
    assert "exit=1" in result


def test_tool_registry_contains_all_new_wrappers():
    for name in [
        "lipsync_state", "osu_status", "content_filter_stats",
        "amika_brain_retrieve", "parse_speech_codes", "filter_input_text",
        "monid_balance", "monid_whoami", "monid_discover", "monid_runs_list",
        "sanitize_speech", "strip_controls", "eval_check_answer",
        "video_probe", "training_emotions", "train_dry_run", "load_rows",
    ]:
        assert name in controller.TOOL_FUNCTIONS, name