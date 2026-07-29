# Model diversity guidance

Updated: 2026-07-23

## Recommendation

Do not assume that four different model IDs will beat four independent samples
from one strong model. For this use case, start with a controlled **2×2 family
roster**: two independent samples from DeepSeek V4 Flash and two from a genuinely
different model family. Compare it against the current four-DeepSeek baseline on
real review tasks before making heterogeneity the default.

A current low-cost, long-context experiment is:

```yaml
ensemble:
  reviewers:
    - model: deepseek/deepseek-v4-flash
      reasoning_effort: high
      temperature: 0.6
    - model: deepseek/deepseek-v4-flash
      reasoning_effort: high
      temperature: 0.6
    - model: google/gemini-3.5-flash-lite
      reasoning_effort: high
      temperature: null
    - model: google/gemini-3.5-flash-lite
      reasoning_effort: high
      temperature: null
```

This preserves two useful sources of variation:

- within-family sampling diversity, measured by separate stochastic OpenCode sessions;
- across-family diversity, which is more likely to expose different blind spots.

At the time of writing, both candidates expose roughly 1M-token contexts on
OpenRouter. DeepSeek V4 Flash supports `xhigh` as its maximum effort, while the
cross-family example deliberately uses the common `high` setting. Gemini's
current ZDR endpoints do not advertise temperature support, so `null` omits it.
Model support and prices change; `review-swarm --check` validates the live roster
before a run. Model pages: <https://openrouter.ai/deepseek/deepseek-v4-flash> and
<https://openrouter.ai/google/gemini-3.5-flash-lite>.

Do not use a model merely to fill a diversity slot. It must have enough context,
reasoning quality, and code/scientific competence for the task. A weak reviewer
can increase false positives and make synthesis harder.

## What the research says

The strongest direct reason for family diversity is error correlation. A 2025
ICML study evaluated more than 350 LLMs and found substantial correlated errors;
on one leaderboard, model pairs agreed 60% of the time when both were wrong.
Shared architecture and provider were among the factors associated with
correlation. Different labels or checkpoints therefore do not guarantee useful
diversity: <https://proceedings.mlr.press/v267/kim25e.html>.

Evidence from multi-agent debate is more cautious. A 2025 systematic study found
that debate often failed to beat Chain-of-Thought or self-consistency despite
more inference compute: <https://arxiv.org/abs/2502.08788>. Another study found
that diversity brought little benefit on its mathematical-reasoning setup but
helped on safety tasks, showing that the value is task-dependent:
<https://arxiv.org/abs/2505.22960>. A 2026 analysis found homogeneous debate can
preserve expected correctness and emphasized diverse initial hypotheses plus
calibrated confidence rather than agent count alone:
<https://arxiv.org/abs/2601.19921>.

Those studies examine debate, while Review Swarm deliberately uses independent
probes followed by evidence-based synthesis. Their failure modes still matter:
more agents and more interaction are not automatically better, majority pressure
can suppress a correct minority, and a strong counterexample matters more than
vote count. The implementation therefore keeps probes isolated and tells the
synthesizer not to use majority rule.

A June 2026 study of requirement-conformance review found another relevant risk:
more detailed prompts increased rejection of correct implementations, and models
were much better at describing failure symptoms than diagnosing root causes. The
new prompts counter this by requiring a concrete trigger or counterexample,
separating verified bugs from hypotheses and ambiguity, and recording attacks
that failed instead of rewarding a high finding count:
<https://link.springer.com/article/10.1007/s10515-026-00638-5>.

## How to decide empirically

Run a small paired evaluation on 10–20 representative past tasks:

1. Four DeepSeek samples (`xhigh`) as the homogeneous baseline.
2. Two DeepSeek plus two Gemini samples (`high`) as the family-diverse arm.
3. Keep evidence, base prompts, job guidance, combiner, and review count fixed.
4. Blindly adjudicate findings against the documents/code or seed known defects.

Track:

- recall of known or human-validated consequential holes;
- unsupported-finding rate;
- unique validated findings contributed by each family;
- pairwise overlap of findings, by model family;
- final-assessment accuracy and usefulness;
- total cost and latency.

Adopt the heterogeneous roster only if its extra unique findings survive
adjudication without an unacceptable false-positive or cost increase. If the
combiner consistently favors its own model family, use a third-family combiner or
rotate the combiner during the evaluation.
