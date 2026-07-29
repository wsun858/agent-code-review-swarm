import asyncio
from pathlib import Path

import pytest

from review_swarm.config import ReviewConfig, ReviewerConfig
from review_swarm.opencode import AgentCall, AgentResult, OpenCodeError
from review_swarm.pipeline import PipelineError, run_pipeline


class FakeRuntime:
    def __init__(self, *, fail_index: int | None = None) -> None:
        self.fail_index = fail_index
        self.reviewer_calls: list[AgentCall] = []
        self.active_reviewers = 0
        self.max_active_reviewers = 0
        self.completed_reviewers = 0
        self.combiner_started_after = -1
        self.combiner_call: AgentCall | None = None

    async def run(self, call: AgentCall) -> AgentResult:
        if call.name == "review-swarm-combiner":
            self.combiner_started_after = self.completed_reviewers
            self.combiner_call = call
            return self._result(
                "# Synthesized Assessment\n**Bottom line:** READY — sound.\n\n"
                "## Most Consequential Holes\nNone"
            )
        self.reviewer_calls.append(call)
        self.active_reviewers += 1
        self.max_active_reviewers = max(
            self.max_active_reviewers, self.active_reviewers
        )
        await asyncio.sleep(0.02)
        self.active_reviewers -= 1
        self.completed_reviewers += 1
        index = int(call.title.rsplit(" ", 1)[1])
        if index == self.fail_index:
            raise OpenCodeError("simulated failure")
        return self._result(f"# Independent Design Probe\nprobe={index}")

    @staticmethod
    def _result(content: str) -> AgentResult:
        return AgentResult(
            content=content,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 20},
                "prompt_tokens_details": {"cached_tokens": 25},
                "total_tokens": 150,
                "cost": 0.001,
            },
            metadata={"session_id": "ses_test", "model": "model"},
            events='{"type":"text"}\n',
        )


def config_for(
    tmp_path: Path,
    *,
    minimum: int | None = None,
    additional_instructions: str | None = None,
) -> ReviewConfig:
    design = tmp_path / "design.md"
    design.write_text("A valid design.", encoding="utf-8")
    return ReviewConfig.model_validate(
        {
            "version": 2,
            "mode": "design-validity",
            "inputs": {"design": design, "requirements": {"text": "Meet the goal."}},
            "ensemble": {"minimum_successful_reviewers": minimum},
            "prompt": {"additional_instructions": additional_instructions},
            "output": {"root": tmp_path / "reviews", "label": "test"},
        }
    )


async def test_reviews_are_parallel_identical_and_combiner_starts_after_all(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    result = await run_pipeline(config_for(tmp_path), runtime)
    assert runtime.max_active_reviewers == 4
    assert runtime.completed_reviewers == 4
    assert runtime.combiner_started_after == 4
    assert len(runtime.reviewer_calls) == 4
    first = runtime.reviewer_calls[0]
    assert all(call.user_prompt == first.user_prompt for call in runtime.reviewer_calls)
    assert all(
        call.system_prompt == first.system_prompt for call in runtime.reviewer_calls
    )
    assert all(call.files == first.files for call in runtime.reviewer_calls)
    assert result.assessment.is_file()
    assessment = result.assessment.read_text(encoding="utf-8")
    assert "750 total" in assessment
    assert "$0.00500000" in assessment
    assert "OpenCode-estimated cost" in assessment


async def test_reviewer_inference_settings_are_applied_independently(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.ensemble.reviewers = [
        ReviewerConfig(
            model="deepseek/deepseek-v4-flash",
            reasoning_effort="high",
            temperature=0.6,
        ),
        ReviewerConfig(
            model="google/gemini-3.5-flash-lite",
            reasoning_effort="high",
            temperature=None,
        ),
    ]
    runtime = FakeRuntime()
    await run_pipeline(config, runtime)
    assert [
        (call.model, call.reasoning_effort, call.temperature)
        for call in runtime.reviewer_calls
    ] == [
        ("deepseek/deepseek-v4-flash", "high", 0.6),
        ("google/gemini-3.5-flash-lite", "high", None),
    ]


async def test_progress_reports_each_agent_and_pipeline_stage(tmp_path: Path) -> None:
    messages: list[str] = []
    await run_pipeline(config_for(tmp_path), FakeRuntime(), progress=messages.append)

    assert any(message.startswith("Run artifacts:") for message in messages)
    assert "Starting 4 reviewers (concurrency 4)" in messages
    assert any(message.startswith("Reviewer 01 started") for message in messages)
    assert any(message.startswith("Reviewer 04 completed") for message in messages)
    assert "Reviewer phase finished: 4/4 succeeded" in messages
    assert "Starting synthesis from 4 reviewer reports" in messages
    assert any(message.startswith("Combiner completed") for message in messages)
    assert any(message.startswith("Assessment written:") for message in messages)


async def test_default_all_success_policy_skips_combiner_on_failure(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(fail_index=1)
    with pytest.raises(PipelineError) as captured:
        await run_pipeline(config_for(tmp_path), runtime)
    assert runtime.completed_reviewers == 4
    assert runtime.combiner_started_after == -1
    assert captured.value.run_dir is not None
    assert (captured.value.run_dir / "run.json").is_file()


async def test_explicit_partial_threshold_allows_combiner(tmp_path: Path) -> None:
    runtime = FakeRuntime(fail_index=1)
    result = await run_pipeline(config_for(tmp_path, minimum=3), runtime)
    assert runtime.combiner_started_after == 4
    assert "PARTIAL ENSEMBLE" in result.assessment.read_text(encoding="utf-8")


async def test_additional_guidance_is_uniform_lower_priority_and_saved(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    result = await run_pipeline(
        config_for(tmp_path, additional_instructions="Probe train/test leakage."),
        runtime,
    )
    reviewer_prompt = runtime.reviewer_calls[0].user_prompt
    assert 'precedence="lower-than-base"' in reviewer_prompt
    assert "Probe train/test leakage." in reviewer_prompt
    assert reviewer_prompt.index("Probe whether") < reviewer_prompt.index(
        "Probe train/test leakage."
    )
    assert all(call.user_prompt == reviewer_prompt for call in runtime.reviewer_calls)
    assert runtime.combiner_call is not None
    assert "Probe train/test leakage." in runtime.combiner_call.user_prompt
    saved = (result.run_dir / "prompts" / "reviewer.txt").read_text(encoding="utf-8")
    assert "Probe train/test leakage." in saved


async def test_secret_scan_blocks_staged_inputs_without_echoing_secret(
    tmp_path: Path,
) -> None:
    secret = "sk-or-v1-abcdefghijklmnopqrstuvwxyz123456"
    config = config_for(tmp_path)
    assert config.inputs.requirements is not None
    config.inputs.requirements.text = f"token={secret}"
    with pytest.raises(PipelineError) as captured:
        await run_pipeline(config, FakeRuntime())
    assert "openrouter_api_key" in str(captured.value)
    assert secret not in str(captured.value)
