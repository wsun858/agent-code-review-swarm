from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_EXCLUDES = [
    ".git/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/venv/**",
    "**/dist/**",
    "**/build/**",
    "**/coverage/**",
    "**/__pycache__/**",
    "**/.pytest_cache/**",
    "**/.mypy_cache/**",
    "**/.ruff_cache/**",
]

DEFAULT_SENSITIVE_PATHS = [
    "**/.env",
    "**/.env.*",
    "**/*.pem",
    "**/*.key",
    "**/.npmrc",
    "**/.pypirc",
    "**/.netrc",
    "**/credentials*",
    "**/.aws/**",
    "**/.ssh/**",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequirementsInput(StrictModel):
    path: Path | None = None
    text: str | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> RequirementsInput:
        if (self.path is None) == (self.text is None):
            raise ValueError(
                "requirements must contain exactly one of 'path' or 'text'"
            )
        if self.text is not None and not self.text.strip():
            raise ValueError("requirements.text cannot be blank")
        return self


class InputsConfig(StrictModel):
    design: Path
    requirements: RequirementsInput | None = None
    workspace: Path | None = None


ReasoningEffort = Literal["xhigh", "high", "medium", "low", "minimal", "none"]


class ReviewerConfig(StrictModel):
    model: str = Field(min_length=1)
    reasoning_effort: ReasoningEffort | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)


def _default_reviewers() -> list[ReviewerConfig]:
    return [
        ReviewerConfig(
            model="deepseek/deepseek-v4-flash",
            reasoning_effort="xhigh",
            temperature=0.6,
        )
        for _ in range(4)
    ]


class EnsembleConfig(StrictModel):
    reviewers: list[ReviewerConfig] = Field(
        default_factory=_default_reviewers, min_length=1, max_length=20
    )
    minimum_successful_reviewers: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_roster(self) -> EnsembleConfig:
        if not all(reviewer.model.strip() for reviewer in self.reviewers):
            raise ValueError("reviewer model cannot be blank")
        minimum = self.minimum_successful_reviewers or self.reviewer_count
        if minimum > self.reviewer_count:
            raise ValueError("minimum_successful_reviewers cannot exceed reviewers")
        return self

    @property
    def reviewer_count(self) -> int:
        return len(self.reviewers)

    @property
    def required_successes(self) -> int:
        return self.minimum_successful_reviewers or self.reviewer_count


class CombinerConfig(StrictModel):
    model: str = "deepseek/deepseek-v4-pro"
    reasoning_effort: ReasoningEffort | None = "xhigh"
    temperature: float | None = Field(default=0.2, ge=0, le=2)


class PromptConfig(StrictModel):
    additional_instructions: str | None = None

    @model_validator(mode="after")
    def reject_blank_instructions(self) -> PromptConfig:
        if self.additional_instructions is not None:
            self.additional_instructions = self.additional_instructions.strip()
            if not self.additional_instructions:
                raise ValueError("prompt.additional_instructions cannot be blank")
        return self


class ContextConfig(StrictModel):
    strategy: Literal["agent"] = "agent"
    include: list[str] = Field(default_factory=lambda: ["**/*"])
    exclude: list[str] = Field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    sensitive_paths: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SENSITIVE_PATHS)
    )


class PrivacyConfig(StrictModel):
    zero_data_retention: bool = True
    data_collection: Literal["allow", "deny"] = "deny"
    secret_scan: bool = True


class OpenCodeConfig(StrictModel):
    executable: str = Field(default="opencode", min_length=1)
    provider: str = Field(default="openrouter", min_length=1)
    concurrency: int = Field(default=4, ge=1, le=20)
    timeout_seconds: float = Field(default=1200, gt=0)


class OutputConfig(StrictModel):
    root: Path = Path("./reviews")
    label: str = "review"
    save_input_bundle: bool = True


class ReviewConfig(StrictModel):
    version: Literal[2] = 2
    mode: Literal["design-validity", "implementation-conformance"]
    inputs: InputsConfig
    context: ContextConfig = Field(default_factory=ContextConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    combiner: CombinerConfig = Field(default_factory=CombinerConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    opencode: OpenCodeConfig = Field(default_factory=OpenCodeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    config_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_mode_inputs(self) -> ReviewConfig:
        if self.mode == "design-validity":
            if self.inputs.requirements is None:
                raise ValueError("design-validity mode requires inputs.requirements")
            if self.inputs.workspace is not None:
                raise ValueError(
                    "design-validity mode does not accept inputs.workspace"
                )
        else:
            if self.inputs.workspace is None:
                raise ValueError(
                    "implementation-conformance mode requires inputs.workspace"
                )
            if self.inputs.requirements is not None:
                raise ValueError(
                    "implementation-conformance mode does not accept inputs.requirements"
                )
        return self


def _resolve_from(base: Path, value: Path) -> Path:
    return (
        value.expanduser().resolve()
        if value.is_absolute()
        else (base / value).resolve()
    )


def load_config(path: str | Path) -> ReviewConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("configuration must be a YAML mapping")
    config = ReviewConfig.model_validate(raw)
    base = config_path.parent
    config.config_path = config_path
    config.inputs.design = _resolve_from(base, config.inputs.design)
    if config.inputs.workspace is not None:
        config.inputs.workspace = _resolve_from(base, config.inputs.workspace)
    if config.inputs.requirements and config.inputs.requirements.path is not None:
        config.inputs.requirements.path = _resolve_from(
            base, config.inputs.requirements.path
        )
    config.output.root = _resolve_from(base, config.output.root)
    return config


def resolved_config_dict(config: ReviewConfig) -> dict:
    data = config.model_dump(mode="json", exclude={"config_path"})
    data["ensemble"]["minimum_successful_reviewers_resolved"] = (
        config.ensemble.required_successes
    )
    return data
