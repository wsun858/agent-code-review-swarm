from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from review_swarm.config import ReviewConfig, load_config


def test_relative_paths_resolve_from_yaml_directory(tmp_path: Path) -> None:
    job = tmp_path / "jobs" / "review.yaml"
    job.parent.mkdir()
    job.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "mode": "design-validity",
                "inputs": {
                    "design": "../docs/design.md",
                    "requirements": {"path": "requirements.md"},
                },
                "output": {"root": "./results"},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(job)
    assert config.inputs.design == (tmp_path / "docs" / "design.md").resolve()
    assert config.inputs.requirements is not None
    assert config.inputs.requirements.path == (job.parent / "requirements.md").resolve()
    assert config.output.root == (job.parent / "results").resolve()


def test_mode_fields_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="does not accept inputs.workspace"):
        ReviewConfig.model_validate(
            {
                "version": 2,
                "mode": "design-validity",
                "inputs": {
                    "design": "design.md",
                    "requirements": {"text": "goals"},
                    "workspace": ".",
                },
            }
        )


def test_default_reviewer_settings_and_strict_success_default() -> None:
    config = ReviewConfig.model_validate(
        {
            "version": 2,
            "mode": "design-validity",
            "inputs": {"design": "design.md", "requirements": {"text": "goals"}},
        }
    )
    assert [reviewer.model for reviewer in config.ensemble.reviewers] == [
        "deepseek/deepseek-v4-flash"
    ] * 4
    assert all(
        reviewer.reasoning_effort == "xhigh"
        for reviewer in config.ensemble.reviewers
    )
    assert all(reviewer.temperature == 0.6 for reviewer in config.ensemble.reviewers)
    assert config.combiner.model == "deepseek/deepseek-v4-pro"
    assert config.ensemble.required_successes == 4


def test_reviewer_list_is_bounded_and_uses_strict_objects() -> None:
    with pytest.raises(ValidationError, match="valid list"):
        ReviewConfig.model_validate(
            {
                "version": 2,
                "mode": "design-validity",
                "inputs": {"design": "design.md", "requirements": {"text": "goals"}},
                "ensemble": {"reviewers": 4},
            }
        )


def test_each_reviewer_has_optional_independent_inference_settings() -> None:
    config = ReviewConfig.model_validate(
        {
            "version": 2,
            "mode": "design-validity",
            "inputs": {"design": "design.md", "requirements": {"text": "goals"}},
            "ensemble": {
                "reviewers": [
                    {
                        "model": "deepseek/deepseek-v4-flash",
                        "reasoning_effort": "high",
                        "temperature": 0.6,
                    },
                    {
                        "model": "google/gemini-3.5-flash-lite",
                        "reasoning_effort": "high",
                        "temperature": None,
                    },
                ]
            },
        }
    )
    assert config.ensemble.reviewer_count == 2
    assert config.ensemble.reviewers[0].temperature == 0.6
    assert config.ensemble.reviewers[1].temperature is None


def test_additional_instructions_are_trimmed_and_bounded() -> None:
    config = ReviewConfig.model_validate(
        {
            "version": 2,
            "mode": "design-validity",
            "inputs": {"design": "design.md", "requirements": {"text": "goals"}},
            "prompt": {"additional_instructions": "  Probe leakage.  "},
        }
    )
    assert config.prompt.additional_instructions == "Probe leakage."

    with pytest.raises(ValidationError, match="cannot be blank"):
        ReviewConfig.model_validate(
            {
                "version": 2,
                "mode": "design-validity",
                "inputs": {
                    "design": "design.md",
                    "requirements": {"text": "goals"},
                },
                "prompt": {"additional_instructions": "   "},
            }
        )
