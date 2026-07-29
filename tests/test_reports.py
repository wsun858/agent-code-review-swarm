import pytest

from review_swarm.reports import aggregate_usage, normalize_usage


def test_usage_accounting_does_not_double_count_reasoning() -> None:
    usage = normalize_usage(
        "reviewer-01",
        "reviewer",
        "model",
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 40},
            "prompt_tokens_details": {"cached_tokens": 80},
            "total_tokens": 150,
            "cost": 0.001,
        },
        1,
    )
    aggregate = aggregate_usage([usage])
    assert aggregate["total"]["total_tokens"] == 150
    assert aggregate["total"]["reasoning_tokens"] == 40
    assert aggregate["total"]["observed_cost"] == pytest.approx(0.001)
