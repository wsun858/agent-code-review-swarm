import json
from pathlib import Path

import pytest

from review_swarm.opencode import (
    AgentCall,
    OpenCodeError,
    _failure_detail,
    _inline_config,
    _isolated_opencode_environment,
    _parse_events,
    _parse_model_catalog,
    _usage_from_steps,
    _validate_model_settings,
)
from review_swarm.config import PrivacyConfig


def test_openrouter_routing_keeps_privacy_without_requiring_sampling_params(
    tmp_path: Path,
) -> None:
    call = AgentCall(
        name="review-swarm-probe",
        title="probe",
        model="google/gemini-3.5-flash-lite",
        reasoning_effort="high",
        temperature=0.6,
        system_prompt="Review.",
        user_prompt="Review.",
        directory=tmp_path,
    )
    config = _inline_config(
        call,
        "openrouter/google/gemini-3.5-flash-lite",
        PrivacyConfig(zero_data_retention=True, data_collection="deny"),
    )
    routing = config["provider"]["openrouter"]["models"][
        "google/gemini-3.5-flash-lite"
    ]["options"]["provider"]
    assert routing == {
        "require_parameters": False,
        "allow_fallbacks": True,
        "zdr": True,
        "data_collection": "deny",
    }


def test_null_inference_settings_are_omitted(tmp_path: Path) -> None:
    call = AgentCall(
        name="review-swarm-probe",
        title="probe",
        model="google/gemini-3.5-flash-lite",
        reasoning_effort=None,
        temperature=None,
        system_prompt="Review.",
        user_prompt="Review.",
        directory=tmp_path,
    )
    config = _inline_config(
        call,
        "openrouter/google/gemini-3.5-flash-lite",
        PrivacyConfig(),
    )
    agent = config["agent"]["review-swarm-probe"]
    assert "variant" not in agent
    assert "temperature" not in agent


def test_model_catalog_and_zdr_preflight_reject_unsupported_temperature() -> None:
    raw = """openrouter/google/gemini-test
{
  "capabilities": {"toolcall": true, "temperature": true},
  "variants": {"high": {"reasoning": {"effort": "high"}}}
}
"""
    catalog = _parse_model_catalog(raw, "openrouter")
    endpoints = [
        {
            "model_id": "google/gemini-test",
            "status": 0,
            "supported_parameters": [
                "max_tokens",
                "tools",
                "tool_choice",
                "reasoning_effort",
            ],
        }
    ]
    _validate_model_settings(
        [("google/gemini-test", "high", None)],
        provider="openrouter",
        catalog=catalog,
        endpoints=endpoints,
    )
    with pytest.raises(OpenCodeError, match="set unsupported values to null"):
        _validate_model_settings(
            [("google/gemini-test", "high", 0.6)],
            provider="openrouter",
            catalog=catalog,
            endpoints=endpoints,
        )


def test_nonzero_exit_reports_structured_stdout_error() -> None:
    raw = json.dumps(
        {
            "type": "error",
            "error": {"name": "APIError", "message": "provider rejected request"},
        }
    )
    detail = _failure_detail(raw, "")
    assert "provider rejected request" in detail


def test_nonzero_exit_falls_back_to_stderr() -> None:
    assert _failure_detail("", "plain stderr failure") == "plain stderr failure"


def test_agent_environment_uses_private_database_and_copies_auth(
    tmp_path: Path,
) -> None:
    shared_data = tmp_path / "shared-data"
    shared_opencode = shared_data / "opencode"
    shared_opencode.mkdir(parents=True)
    auth = shared_opencode / "auth.json"
    auth.write_text(
        '{"openrouter":{"type":"api","key":"connected-key"}}',
        encoding="utf-8",
    )
    environment = {
        "HOME": str(tmp_path / "home"),
        "XDG_DATA_HOME": str(shared_data),
        "OPENROUTER_API_KEY": "test-key",
    }

    with _isolated_opencode_environment(
        environment, provider="openrouter"
    ) as isolated:
        private_data = Path(isolated["XDG_DATA_HOME"])
        assert private_data != shared_data
        copied_auth = private_data / "opencode" / "auth.json"
        assert json.loads(copied_auth.read_text()) == json.loads(auth.read_text())
        assert isolated["OPENROUTER_API_KEY"] == "test-key"
        assert "XDG_DATA_HOME" in environment

    assert not private_data.exists()


def test_environment_key_seeds_private_auth_file(tmp_path: Path) -> None:
    environment = {
        "HOME": str(tmp_path),
        "OPENROUTER_API_KEY": "environment-key",
    }
    with _isolated_opencode_environment(
        environment, provider="openrouter"
    ) as isolated:
        auth_path = Path(isolated["XDG_DATA_HOME"]) / "opencode" / "auth.json"
        credentials = json.loads(auth_path.read_text(encoding="utf-8"))
        assert credentials == {
            "openrouter": {"type": "api", "key": "environment-key"}
        }
        assert auth_path.stat().st_mode & 0o777 == 0o600


def test_agent_environments_are_unique(tmp_path: Path) -> None:
    environment = {"HOME": str(tmp_path)}
    with _isolated_opencode_environment(environment) as first:
        with _isolated_opencode_environment(environment) as second:
            assert first["XDG_DATA_HOME"] != second["XDG_DATA_HOME"]


def test_json_events_yield_final_text_and_usage() -> None:
    raw = "\n".join(
        [
            json.dumps(
                {
                    "type": "step_finish",
                    "sessionID": "ses_1",
                    "part": {
                        "tokens": {
                            "total": 150,
                            "input": 75,
                            "output": 30,
                            "reasoning": 20,
                            "cache": {"read": 25, "write": 0},
                        },
                        "cost": 0.001,
                    },
                }
            ),
            json.dumps(
                {
                    "type": "text",
                    "sessionID": "ses_1",
                    "part": {"text": "# Independent Design Probe\nResult"},
                }
            ),
        ]
    )
    result = _parse_events(raw, model="openrouter/model", stderr="")
    assert result.content.startswith("# Independent Design Probe")
    assert result.metadata["session_id"] == "ses_1"
    assert result.usage["prompt_tokens"] == 100
    assert result.usage["completion_tokens"] == 50
    assert result.usage["total_tokens"] == 150
    assert result.usage["cost"] == pytest.approx(0.001)


def test_missing_final_text_fails() -> None:
    raw = json.dumps({"type": "error", "error": {"name": "APIError"}})
    with pytest.raises(OpenCodeError, match="no final text"):
        _parse_events(raw, model="openrouter/model", stderr="")


def test_usage_falls_back_when_provider_omits_total() -> None:
    usage = _usage_from_steps(
        [
            {
                "tokens": {
                    "input": 10,
                    "output": 5,
                    "reasoning": 3,
                    "cache": {"read": 2, "write": 1},
                },
                "cost": 0.2,
            }
        ]
    )
    assert usage["total_tokens"] == 21
