from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from .config import OpenCodeConfig, PrivacyConfig


class OpenCodeError(RuntimeError):
    """Raised when an OpenCode agent session cannot produce a result."""


@dataclass(frozen=True)
class AgentCall:
    name: str
    title: str
    model: str
    reasoning_effort: str | None
    temperature: float | None
    system_prompt: str
    user_prompt: str
    directory: Path
    files: tuple[Path, ...] = ()
    denied_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentResult:
    content: str
    usage: dict[str, Any]
    metadata: dict[str, Any]
    events: str


class OpenCodeRuntime:
    """Run isolated OpenCode CLI sessions and parse their JSON event streams."""

    def __init__(
        self,
        config: OpenCodeConfig,
        privacy: PrivacyConfig,
        *,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.privacy = privacy
        self.environment = environment

    def executable(self) -> str:
        resolved = shutil.which(self.config.executable)
        if resolved is None:
            raise OpenCodeError(
                f"OpenCode executable not found: {self.config.executable!r}; "
                "install it from https://opencode.ai/docs/ or set "
                "opencode.executable"
            )
        return resolved

    async def version(self) -> str:
        process = await asyncio.create_subprocess_exec(
            self.executable(),
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise OpenCodeError(f"cannot run OpenCode: {detail or process.returncode}")
        return stdout.decode("utf-8", errors="replace").strip()

    async def run(self, call: AgentCall) -> AgentResult:
        model = _model_id(self.config.provider, call.model)
        inline_config = _inline_config(call, model, self.privacy)
        env = dict(os.environ if self.environment is None else self.environment)
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(inline_config)
        # Reviewed repositories are evidence. Their AGENTS.md/CLAUDE.md and
        # project OpenCode config must not become higher-priority instructions.
        env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "true"
        env["OPENCODE_DISABLE_CLAUDE_CODE_PROMPT"] = "true"

        command = [
            self.executable(),
            "run",
            "--format",
            "json",
            "--agent",
            call.name,
            "--model",
            model,
            "--dir",
            str(call.directory),
            "--title",
            call.title,
        ]
        if call.reasoning_effort is not None:
            command.extend(["--variant", call.reasoning_effort])
        for path in call.files:
            command.extend(["--file", str(path)])

        # Every `opencode run` writes session state to opencode.db. Concurrent
        # CLI processes otherwise race on that global SQLite database before a
        # model request is made. Isolate data (and therefore the database) per
        # invocation while retaining global config and provider credentials.
        with _isolated_opencode_environment(
            env, provider=self.config.provider
        ) as isolated_env:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=isolated_env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(call.user_prompt.encode("utf-8")),
                    timeout=self.config.timeout_seconds,
                )
            except TimeoutError as exc:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                raise OpenCodeError(
                    f"OpenCode session timed out after "
                    f"{self.config.timeout_seconds:g}s"
                ) from exc

            raw = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace").strip()
            if process.returncode != 0:
                raise OpenCodeError(
                    f"OpenCode exited with status {process.returncode}: "
                    f"{_failure_detail(raw, err)}"
                )
            return _parse_events(raw, model=model, stderr=err)

    async def validate_models(
        self,
        settings: Sequence[tuple[str, str | None, float | None]],
    ) -> None:
        """Fail before paid calls when configured model settings are unsupported."""
        process = await asyncio.create_subprocess_exec(
            self.executable(),
            "models",
            self.config.provider,
            "--verbose",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise OpenCodeError(
                f"cannot inspect OpenCode model capabilities: "
                f"{detail or process.returncode}"
            )
        catalog = _parse_model_catalog(
            stdout.decode("utf-8", errors="replace"), self.config.provider
        )
        endpoints: list[dict[str, Any]] | None = None
        if self.config.provider == "openrouter" and self.privacy.zero_data_retention:
            endpoints = await asyncio.to_thread(_fetch_openrouter_zdr_endpoints)
        _validate_model_settings(
            settings,
            provider=self.config.provider,
            catalog=catalog,
            endpoints=endpoints,
        )


@contextmanager
def _isolated_opencode_environment(
    environment: dict[str, str],
    *,
    provider: str | None = None,
) -> Iterator[dict[str, str]]:
    """Give one OpenCode process a private database and copied credentials."""
    isolated = dict(environment)
    source_data_home = environment.get("XDG_DATA_HOME")
    if source_data_home:
        shared_opencode_dir = Path(source_data_home) / "opencode"
    else:
        home = Path(environment.get("HOME", str(Path.home())))
        shared_opencode_dir = home / ".local" / "share" / "opencode"

    with tempfile.TemporaryDirectory(prefix="review-swarm-opencode-") as temp_dir:
        private_data_home = Path(temp_dir)
        private_opencode_dir = private_data_home / "opencode"
        private_opencode_dir.mkdir()
        _seed_auth_file(
            shared_opencode_dir / "auth.json",
            private_opencode_dir / "auth.json",
            environment,
            provider,
        )
        isolated["XDG_DATA_HOME"] = str(private_data_home)
        yield isolated


def _seed_auth_file(
    source: Path,
    destination: Path,
    environment: dict[str, str],
    provider: str | None,
) -> None:
    credentials: dict[str, Any] = {}
    if source.is_file():
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                credentials.update(value)
        except (OSError, json.JSONDecodeError):
            pass

    if provider and provider not in credentials:
        key = next(
            (
                environment[name]
                for name in _provider_key_names(provider)
                if environment.get(name)
            ),
            None,
        )
        if key:
            credentials[provider] = {"type": "api", "key": key}

    if credentials:
        destination.write_text(json.dumps(credentials), encoding="utf-8")
        destination.chmod(0o600)


def _provider_key_names(provider: str) -> tuple[str, ...]:
    known = {
        "anthropic": ("ANTHROPIC_API_KEY",),
        "google": ("GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"),
        "openai": ("OPENAI_API_KEY",),
        "openrouter": ("OPENROUTER_API_KEY",),
    }
    normalized = "".join(
        character if character.isalnum() else "_" for character in provider
    )
    fallback = normalized.upper() + "_API_KEY"
    return known.get(provider, (fallback,))


def _model_id(provider: str, model: str) -> str:
    prefix = provider.rstrip("/") + "/"
    return model if model.startswith(prefix) else prefix + model


def _failure_detail(raw: str, stderr: str) -> str:
    """Extract OpenCode's structured stdout error before falling back to stderr."""
    errors: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "error":
            errors.append(event)
    if errors:
        return json.dumps(errors[-1], ensure_ascii=False)[-4000:]
    if stderr:
        return stderr[-4000:]
    if raw.strip():
        return raw.strip()[-4000:]
    return "no error output"


def _parse_model_catalog(raw: str, provider: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    lines = raw.splitlines()
    prefix = provider.rstrip("/") + "/"
    for index, line in enumerate(lines):
        model = line.strip()
        if not model.startswith(prefix) or index + 1 >= len(lines):
            continue
        try:
            value = json.loads(lines[index + 1])
        except json.JSONDecodeError:
            block: list[str] = []
            depth = 0
            for candidate in lines[index + 1 :]:
                block.append(candidate)
                depth += candidate.count("{") - candidate.count("}")
                if depth == 0 and block:
                    break
            try:
                value = json.loads("\n".join(block))
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict):
            records[model] = value
    return records


def _fetch_openrouter_zdr_endpoints() -> list[dict[str, Any]]:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/endpoints/zdr",
        headers={"User-Agent": "review-swarm/0.3"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise OpenCodeError(
            f"cannot validate OpenRouter ZDR endpoint availability: {exc}"
        ) from exc
    values = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        raise OpenCodeError("invalid OpenRouter ZDR endpoint response")
    return [value for value in values if isinstance(value, dict)]


def _validate_model_settings(
    settings: Sequence[tuple[str, str | None, float | None]],
    *,
    provider: str,
    catalog: dict[str, dict[str, Any]],
    endpoints: list[dict[str, Any]] | None,
) -> None:
    for model, reasoning_effort, temperature in settings:
        full_model = _model_id(provider, model)
        metadata = catalog.get(full_model)
        if metadata is None:
            raise OpenCodeError(f"model is not available in OpenCode: {full_model}")
        capabilities = metadata.get("capabilities", {})
        if not isinstance(capabilities, dict) or not capabilities.get("toolcall"):
            raise OpenCodeError(
                f"model does not support required tool calls: {full_model}"
            )
        if reasoning_effort is not None:
            variants = metadata.get("variants", {})
            if not isinstance(variants, dict) or reasoning_effort not in variants:
                available = (
                    ", ".join(sorted(variants))
                    if isinstance(variants, dict)
                    else ""
                )
                raise OpenCodeError(
                    f"model {full_model} does not support reasoning_effort "
                    f"{reasoning_effort!r}; available: {available or 'none'}"
                )
        if temperature is not None and not capabilities.get("temperature"):
            raise OpenCodeError(
                f"model does not support temperature: {full_model}; set it to null"
            )
        if endpoints is not None:
            _validate_openrouter_route(
                model.removeprefix(provider.rstrip("/") + "/"),
                reasoning_effort,
                temperature,
                endpoints,
            )


def _validate_openrouter_route(
    model: str,
    reasoning_effort: str | None,
    temperature: float | None,
    endpoints: list[dict[str, Any]],
) -> None:
    candidates = [
        endpoint
        for endpoint in endpoints
        if endpoint.get("model_id") == model and endpoint.get("status") == 0
    ]
    if not candidates:
        raise OpenCodeError(f"no active ZDR endpoint is available for openrouter/{model}")

    def compatible(endpoint: dict[str, Any]) -> bool:
        supported = endpoint.get("supported_parameters", [])
        if not isinstance(supported, list):
            return False
        required = {"max_tokens", "tools", "tool_choice"}
        if temperature is not None:
            required.add("temperature")
        if not required.issubset(supported):
            return False
        return reasoning_effort is None or (
            "reasoning" in supported or "reasoning_effort" in supported
        )

    if not any(compatible(endpoint) for endpoint in candidates):
        requested = ["tools"]
        if reasoning_effort is not None:
            requested.append(f"reasoning_effort={reasoning_effort}")
        if temperature is not None:
            requested.append(f"temperature={temperature:g}")
        raise OpenCodeError(
            f"no active ZDR endpoint for openrouter/{model} supports configured "
            f"parameters ({', '.join(requested)}); set unsupported values to null "
            "or choose another model"
        )


def _inline_config(
    call: AgentCall, model: str, privacy: PrivacyConfig
) -> dict[str, Any]:
    provider, model_id = model.split("/", 1)
    provider_options: dict[str, Any] = {}
    if provider == "openrouter":
        provider_options.update(
            {
                # OpenCode emits family defaults such as top_k/top_p even when
                # an OpenRouter endpoint does not advertise them. Requiring
                # every parameter would filter out otherwise compatible
                # endpoints (notably Gemini 3.5 Flash Lite's ZDR endpoints).
                "require_parameters": False,
                "allow_fallbacks": True,
                "zdr": privacy.zero_data_retention,
                "data_collection": privacy.data_collection,
            }
        )
    read_permission: str | dict[str, str]
    if call.denied_paths:
        read_permission = {"*": "allow"}
        read_permission.update({pattern: "deny" for pattern in call.denied_paths})
    else:
        read_permission = "allow"
    agent_config: dict[str, Any] = {
        "description": "Read-only independent review-swarm agent",
        "mode": "primary",
        "model": model,
        "prompt": call.system_prompt,
        "permission": {
            "*": "deny",
            "read": read_permission,
            "glob": "allow",
            "grep": "allow",
            "list": "allow",
            "lsp": "allow",
            "doom_loop": "deny",
            "external_directory": "deny",
        },
    }
    if call.reasoning_effort is not None:
        agent_config["variant"] = call.reasoning_effort
    if call.temperature is not None:
        agent_config["temperature"] = call.temperature
    return {
        "$schema": "https://opencode.ai/config.json",
        "share": "disabled",
        "snapshot": False,
        "agent": {call.name: agent_config},
        "provider": {
            provider: {
                "models": {
                    model_id: (
                        {"options": {"provider": provider_options}}
                        if provider_options
                        else {}
                    ),
                }
            }
        },
    }


def _parse_events(raw: str, *, model: str, stderr: str) -> AgentResult:
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OpenCodeError(f"invalid OpenCode JSON event: {line[:200]}") from exc
        if isinstance(value, dict):
            events.append(value)

    texts = [
        event.get("part", {}).get("text", "").strip()
        for event in events
        if event.get("type") == "text"
        and isinstance(event.get("part"), dict)
        and isinstance(event["part"].get("text"), str)
        and event["part"].get("text", "").strip()
    ]
    errors = [event for event in events if event.get("type") == "error"]
    if not texts:
        detail = json.dumps(errors[-1], ensure_ascii=False) if errors else stderr
        raise OpenCodeError(
            f"OpenCode returned no final text: {detail or 'unknown error'}"
        )

    steps = [
        event["part"]
        for event in events
        if event.get("type") == "step_finish" and isinstance(event.get("part"), dict)
    ]
    usage = _usage_from_steps(steps)
    session_id = next(
        (event.get("sessionID") for event in events if event.get("sessionID")), None
    )
    metadata = {
        "session_id": session_id,
        "model": model,
        "steps": len(steps),
        "stderr": stderr or None,
    }
    return AgentResult(texts[-1], usage, metadata, raw)


def _usage_from_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    cache_read = 0
    cache_write = 0
    total_tokens = 0
    cost = 0.0
    for step in steps:
        tokens = step.get("tokens") if isinstance(step.get("tokens"), dict) else {}
        cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
        current_input = _number(tokens.get("input"))
        current_output = _number(tokens.get("output"))
        current_reasoning = _number(tokens.get("reasoning"))
        current_read = _number(cache.get("read"))
        current_write = _number(cache.get("write"))
        input_tokens += current_input
        output_tokens += current_output
        reasoning_tokens += current_reasoning
        cache_read += current_read
        cache_write += current_write
        total_tokens += _number(tokens.get("total")) or (
            current_input
            + current_output
            + current_reasoning
            + current_read
            + current_write
        )
        if isinstance(step.get("cost"), (int, float)):
            cost += float(step["cost"])
    return {
        "prompt_tokens": input_tokens + cache_read + cache_write,
        "completion_tokens": output_tokens + reasoning_tokens,
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "prompt_tokens_details": {
            "cached_tokens": cache_read,
            "cache_write_tokens": cache_write,
        },
        "total_tokens": total_tokens,
        "cost": cost,
    }


def _number(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0
