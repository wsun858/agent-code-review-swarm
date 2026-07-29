from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

from . import __version__
from .config import load_config
from .opencode import OpenCodeError, OpenCodeRuntime
from .pipeline import PipelineError, prepare_inputs, run_pipeline


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="review-swarm",
        description="Run independent OpenCode reviews and synthesize a local assessment.",
    )
    result.add_argument("config", type=Path, nargs="?", help="path to review YAML")
    result.add_argument(
        "--check",
        action="store_true",
        help="validate configuration, inputs, and OpenCode without running agents",
    )
    result.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.config is None:
        parser().error("the following arguments are required: config")
    try:
        config = load_config(args.config)
        runtime = OpenCodeRuntime(config.opencode, config.privacy)
        if args.check:
            with tempfile.TemporaryDirectory(prefix="review-swarm-check-") as temporary:
                inputs = Path(temporary) / "inputs"
                inputs.mkdir()
                prepared = prepare_inputs(config, Path(temporary))
            opencode_version = asyncio.run(runtime.version())
            settings = [
                (reviewer.model, reviewer.reasoning_effort, reviewer.temperature)
                for reviewer in config.ensemble.reviewers
            ] + [
                (
                    config.combiner.model,
                    config.combiner.reasoning_effort,
                    config.combiner.temperature,
                )
            ]
            asyncio.run(runtime.validate_models(settings))
            print(f"Configuration valid: {config.config_path}")
            print(f"Mode: {config.mode}")
            print(f"Input hashes: {prepared.hashes}")
            print(f"OpenCode: {opencode_version}")
            print(f"Output root: {config.output.root}")
            return 0
        print(
            f"[review-swarm] Loaded {config.mode} job: "
            f"{config.ensemble.reviewer_count} reviewers + 1 combiner",
            file=sys.stderr,
            flush=True,
        )
        result = asyncio.run(
            run_pipeline(
                config,
                runtime,
                progress=lambda message: print(
                    f"[review-swarm] {message}", file=sys.stderr, flush=True
                ),
            )
        )
        print(f"Assessment: {result.assessment}")
        print(f"Run artifacts: {result.run_dir}")
        total = result.usage["total"]
        print(f"Tokens: {total['total_tokens']:,}")
        print("OpenCode-estimated cost by agent:")
        for line in _agent_cost_lines(result.usage):
            print(f"  {line}")
        print(f"Estimated total cost: ${total['observed_cost']:.8f}")
        return 0
    except (
        OSError,
        TypeError,
        ValueError,
        ValidationError,
        OpenCodeError,
        RuntimeError,
    ) as exc:
        if isinstance(exc, PipelineError) and exc.run_dir:
            print(f"Run artifacts: {exc.run_dir}", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _agent_cost_lines(usage: dict) -> list[str]:
    lines = []
    for entry in usage["calls"]:
        if entry["category"] == "reviewer":
            label = entry["call"].replace("reviewer-", "Reviewer ")
        else:
            label = "Combiner"
        cost = entry.get("cost")
        amount = f"${cost:.8f}" if isinstance(cost, (int, float)) else "unavailable"
        lines.append(f"{label}: {amount}")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
