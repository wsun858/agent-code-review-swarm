from review_swarm.cli import _agent_cost_lines


def test_agent_cost_lines_include_each_call_and_missing_costs() -> None:
    usage = {
        "calls": [
            {"call": "reviewer-01", "category": "reviewer", "cost": 0.0123},
            {"call": "reviewer-02", "category": "reviewer", "cost": None},
            {"call": "combiner", "category": "combiner", "cost": 0.0456},
        ]
    }

    assert _agent_cost_lines(usage) == [
        "Reviewer 01: $0.01230000",
        "Reviewer 02: unavailable",
        "Combiner: $0.04560000",
    ]
