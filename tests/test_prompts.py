from importlib.resources import files


def prompt(name: str) -> str:
    return files("review_swarm.prompts").joinpath(name).read_text(encoding="utf-8")


def test_mode_prompts_are_adversarial_and_research_oriented() -> None:
    design = prompt("design_validity.txt")
    implementation = prompt("implementation_conformance.txt")
    for content in [design, implementation]:
        assert "expert research scientist" in content
        assert "counterexample" in content or "concrete trigger" in content
        assert "What Resisted Probing" in content
        assert "Matrix" not in content
    assert "requirements checklist" in design
    assert "mechanical design-commitment checklist" in implementation
