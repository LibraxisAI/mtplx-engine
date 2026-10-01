from mtplx.commands import public
from mtplx.commands.trace import _match_receipt


def test_profile_advertises_current_vision_and_replays_reasoning():
    args = dict(model_id="custom-model", base_url="http://localhost:8000/v1",
                api_key="local", workspace_path="/workspace")
    vision = public._hermes_config_yaml(**args, vision=True)
    assert "supports_vision: true" in vision
    assert "reasoning_echo: true" in vision
    assert "tool_image_retention: until_compaction" in vision
    text = public._hermes_merged_config_yaml(vision, public._hermes_config_yaml(**args, vision=False))
    assert "supports_vision: false" in text
    assert "supports_vision: true" not in text


def test_hermes_trace_rejects_ambiguous_or_unrelated_receipts():
    message = {"_client": "hermes", "_hermes_api_counts": [15688, 22097],
               "time": {"completed": 100000}}
    receipt = {"logged_at_s": 99.8, "prompt_tokens": 15688, "completion_tokens": 22097}
    wrong = {**receipt, "completion_tokens": 30}
    assert _match_receipt(message, [wrong, receipt], set()) is receipt
    assert _match_receipt(message, [receipt, dict(receipt)], set()) is None
    assert _match_receipt(message, [{**receipt, "logged_at_s": 80}], set()) is None


def test_hermes_dotenv_leaves_the_working_directory_to_config_yaml():
    """Hermes v0.21 deprecates TERMINAL_CWD in .env (a warning on every launch);
    terminal.cwd in config.yaml is the canonical setting and both writers emit it."""
    from mtplx.commands import public

    env_text = public._hermes_dotenv(
        model_id="mtplx-flash-next-optimized-speed",
        base_url="http://127.0.0.1:8000/v1",
        api_key="mtplx-local",
        workspace_path="/tmp/work space",
    )
    assert "TERMINAL_CWD" not in env_text
    assert 'HERMES_WORKSPACE="/tmp/work space"' in env_text
    config_text = public._hermes_config_yaml(
        model_id="mtplx-flash-next-optimized-speed",
        base_url="http://127.0.0.1:8000/v1",
        api_key="mtplx-local",
        workspace_path="/tmp/work space",
    )
    terminal = config_text.split("terminal:\n", 1)[1].split("\ncompression:", 1)[0]
    assert "cwd: '/tmp/work space'" in terminal
