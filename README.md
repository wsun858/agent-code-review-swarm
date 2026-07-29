# Review Swarm

Review Swarm runs a five-agent OpenCode review: four isolated reviewers execute
concurrently, then a fifth agent synthesizes their completed reports. It supports:

- design validity: a design document against requirements or a raw prompt;
- implementation conformance: a design document against a repository that each
  reviewer explores with OpenCode's read-only tools.

OpenCode owns model access, repository exploration, tool-result pruning, and
context compaction. The small Python runner owns only deterministic orchestration
and local artifacts; it never calls a model API directly.

## Install

Install and authenticate OpenCode first. For the default OpenRouter model, either
use OpenCode's `/connect` flow or export `OPENROUTER_API_KEY` as usual. See the
[OpenCode provider guide](https://opencode.ai/docs/providers#openrouter).
This port is tested with OpenCode 1.18.4.

Then install this project:

```console
git clone <repository-url>
cd review-swarm
uv sync --extra dev
```

Validate a job without running paid agents:

```console
uv run review-swarm /absolute/path/to/review.yaml --check
```

Run it:

```console
uv run review-swarm /absolute/path/to/review.yaml
```

## How the five agents connect

The runner starts four independent commands concurrently:

```text
opencode run --format json --agent review-swarm-probe ...
```

Every reviewer receives the same source-controlled system prompt, the same
mode-specific prompt, and the same input attachments. Implementation reviewers
start in the target repository and use only read/list/glob/grep/LSP tools; edits,
shell execution, web access, nested agents, and external-directory reads are
denied. Each process is a separate OpenCode session with a private temporary data
home and SQLite database, so parallel reviewers neither contend on OpenCode's
global database nor see one another. Connected-provider credentials are copied
into each temporary data home. Environment-based API keys are securely seeded as
temporary OpenCode credentials because OpenCode 1.18.4 does not apply them when
initializing a fresh data home. The temporary homes are removed after each agent.

The runner waits for the configured success threshold and saves each final answer
as `reviewers/01.md`, `02.md`, and so on. It then starts exactly one combiner
session, attaching all successful reviewer files plus the original design and
requirements. For implementation reviews, the combiner also starts in the target
repository so it can verify disputed claims. Thus reviewer output lands in the
combiner automatically and synthesis cannot start early.

This process-level fan-out is intentionally used instead of OpenCode's native
background subagents. Native background subagents are still experimental; the
batch runner gives this workflow deterministic cardinality, true concurrency,
clear failure thresholds, and a hard fan-in barrier while retaining OpenCode's
context management.

## Configure a review

Copy [`review.example.yaml`](review.example.yaml).
Relative paths resolve from the YAML file's directory.

For design validity:

```yaml
mode: design-validity
inputs:
  design: ./design.md
  requirements:
    path: ./requirements.md
    # Or: text: |-
    #   The original prompt goes here.
```

For implementation conformance:

```yaml
mode: implementation-conformance
inputs:
  design: ./design.md
  workspace: ../implementation
```

Configuration version 2 represents every reviewer explicitly so heterogeneous
models do not inherit misleading inference settings:

```yaml
ensemble:
  reviewers:
    - model: deepseek/deepseek-v4-flash
      reasoning_effort: high
      temperature: 0.6
    - model: google/gemini-3.5-flash-lite
      reasoning_effort: high
      temperature: null  # omitted; use the model/provider default
```

`model` is required; `reasoning_effort` and `temperature` are optional. A null
value is not sent to OpenCode. Before any paid call, the runner checks that every
model exists, supports tool calls and the requested reasoning variant, and has an
active privacy-compatible OpenRouter endpoint for each explicitly configured
parameter. `--check` performs the same live preflight. The default combiner is
`deepseek/deepseek-v4-pro` at `xhigh` reasoning and temperature `0.2`. Review
Swarm does not impose a token or cost budget; model and provider limits apply.

While a run is active, the terminal reports each reviewer as it starts, emits a
heartbeat every 15 seconds, reports completion or failure, and then reports
synthesis progress.

`context.include` and `context.exclude` guide implementation exploration rather
than constructing a fixed source snapshot. Sensitive paths are explicitly listed
in every reviewer prompt and tool access is confined to the workspace.

Task-specific guidance is still wrapped below the fixed prompts and supplied
identically to all five agents:

```yaml
prompt:
  additional_instructions: |-
    Probe especially for train/test leakage.
```

## Outputs

Each run creates a unique directory:

```text
reviews/<timestamp>-<label>-<hash>/
├── assessment.md
├── run.json
├── usage.json
├── usage.csv
├── config.resolved.yaml
├── inputs/
├── prompts/
├── reviewers/
└── api/                 # OpenCode JSON event streams and session metadata
```

Token counts and reasoning/cache breakdowns come from OpenCode's structured
events. Cost is OpenCode's model-catalog estimate, not the provider's final
charged amount; consult OpenRouter billing for authoritative cost. The final
terminal summary prints the estimate for every reviewer and the combiner before
printing the aggregate estimate.
