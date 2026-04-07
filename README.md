# Complaint LLM-as-a-Judge

An automated, fully decoupled evaluation system that grades the quality of **SCRIBE** complaint summaries using local LLMs as judges. Every evaluation rule, model choice, threshold, and routing policy is driven by YAML config files — no hardcoding.

---

## What This System Does

When a customer files a complaint, two upstream systems process it:

| System | Role | Output |
|--------|------|--------|
| **SCRIBE** | Reads the raw complaint and writes a concise summary for support agents | `scribe_summary` |
| **TRACE** | Classifies the complaint into a category | `trace_category` |

**This system evaluates SCRIBE's summary quality only.** It asks: *"Is this summary faithful, complete, clear, and action-ready?"* — using LLMs as the evaluators (judges).

---

## Architecture Overview

```
cases.json          ← complaint data (add new cases here — no code needed)
      │
      ▼
policies.yaml       ← keyword + severity routing → selects evaluation "rulebook"
      │
      ▼
evaluation_profiles.yaml  ← the rulebook: which metrics run, which models, what thresholds
      │
      ▼
judge.py            ← calls local Ollama LLMs, scores each criterion, applies verdict
      │
      ▼
PASS / NEEDS_REVIEW / FAIL  +  detailed audit report
```

Everything that drives the system's decisions lives in the three data files. The Python scripts are just the execution engine.

---

## File Structure

```
complaint_judge/
├── cases.json                       # Test complaint cases (add new ones here)
├── cases.py                         # Thin loader — reads cases.json
├── judge.py                         # Evaluation engine: calls LLMs, scores, verdicts
├── profile_loader.py                # Reads YAML configs, resolves per-complaint profile
├── run.py                           # Orchestrator: runs all cases, prints report, saves audit
├── requirements.txt
├── README.md
└── config/
    ├── policies.yaml                # Business routing: conditions → eval profile
    └── evaluation_profiles.yaml    # Eval profiles: criteria, models, weights, thresholds
```

---

## The Three Config Files

### `cases.json` — Input Data

Each entry is one complaint that went through SCRIBE and TRACE:

```json
{
    "id": "COMP-001",
    "severity":       "low",
    "department":     "billing",
    "complaint":      "I was charged a $35 overdraft fee on March 28th...",
    "scribe_summary": "Customer charged $35 overdraft fee on March 28...",
    "trace_category": "Billing/Fees Dispute",

    "overrides": {
        "entity_preservation": { "enabled": false },
        "faithfulness": { "judge_model": "gemma3:4b", "weight": 2.0 }
    }
}
```

The optional `overrides` block lets you **patch any profile setting for that one specific case** — turn a metric off, switch to a different model, change a weight — without touching the shared profile.

**To add a new complaint: just add a JSON block here. No Python changes.**

---

### `config/policies.yaml` — The Traffic Cop

Determines which evaluation "rulebook" (profile) to apply to each complaint. Rules are checked **top to bottom — first match wins**.

```yaml
default_profile: standard

policies:
  - name: fraud_complaints
    description: "Fraud, theft, or unauthorized charge complaints"
    conditions:
      keywords: ["fraud", "unauthorized", "identity theft", "chargeback"]
    eval_profile: fraud_focused

  - name: high_risk_complaints
    description: "Regulatory or legal language detected"
    conditions:
      keywords: ["CFPB", "regulator", "legal action", "violation"]
    eval_profile: stringent

  - name: high_severity
    conditions:
      severity: high
    eval_profile: stringent

  - name: loan_complaints
    conditions:
      keywords: ["loan", "mortgage", "home equity", "foreclosure"]
    eval_profile: loan_focused
```

**Condition types:**
- `keywords` — OR-matched against the complaint text (case-insensitive). If any keyword appears anywhere in the complaint → match.
- `severity` — matched against `case["severity"]` field.
- Multiple conditions in one policy are AND-ed.

**To add a new routing rule: add a block here. No Python changes.**

---

### `config/evaluation_profiles.yaml` — The Rulebook

Defines exactly how to evaluate a SCRIBE summary — which metrics to run, which model judges each one, and how strict to be.

```yaml
fraud_focused:
  pass_threshold: 0.78   # weighted avg must be >= this to PASS
  fail_threshold: 0.55   # weighted avg < this = FAIL (also per-criterion)

  criteria:
    faithfulness:
      enabled: true
      judge_model: gemma3:4b   # stricter model for fraud fact-checking
      weight: 3.0              # triple-weighted — hallucinated amounts are dangerous
      pass_threshold: 0.80     # this criterion needs >= 0.80
      fail_threshold: 0.60     # if < 0.60 → hard FAIL regardless of avg
```

#### The Four Profiles

| Profile | Used when | Pass ≥ | Fail < | Criteria |
|---------|-----------|:------:|:------:|----------|
| `standard` | General / low-risk complaints | 0.75 | 0.50 | 6 of 7 (urgency_tone off) |
| `stringent` | High-risk, regulatory, compliance | 0.80 | 0.60 | All 7 |
| `fraud_focused` | Fraud / unauthorized transactions | 0.78 | 0.55 | 6 of 7 (conciseness off) |
| `loan_focused` | Loan / mortgage complaints | 0.75 | 0.50 | All 7 |

#### The Seven SCRIBE Evaluation Criteria

| Criterion | What it checks |
|-----------|----------------|
| `faithfulness` | Did SCRIBE invent anything not in the original complaint? (hallucination check) |
| `coverage` | Did SCRIBE capture the problem, customer impact, urgency, and what they want? |
| `action_orientedness` | Can a support agent immediately know what to do from the summary alone? |
| `conciseness` | Is the summary appropriately concise, or did SCRIBE over-explain? |
| `clarity` | Is the summary easy to understand in 5 seconds? |
| `urgency_tone` | Does the summary preserve the urgency and frustration from the original complaint? |
| `entity_preservation` | Are key amounts, dates, reference numbers, and IDs retained in the summary? |

---

## Scoring and Verdict Logic

### How scores are calculated

Each enabled criterion gets a **raw score of 1–5** from the LLM judge, normalized to **0.0–1.0**. A **weighted average** is then computed across all enabled criteria:

```
weighted_avg = Σ(score × weight) / Σ(weights)
```

### Verdict: Option B — Strict

```
PASS         → weighted_avg >= profile.pass_threshold
               AND no individual criterion below its fail_threshold

NEEDS_REVIEW → weighted_avg is between pass and fail
               AND no criterion gate breached

FAIL         → weighted_avg < profile.fail_threshold
               OR any single criterion below its fail_threshold
```

The strict rule means: **even a good overall average is overridden if one critical dimension fails its gate.** For example, a fraud case with excellent coverage but a hallucinated amount will FAIL on faithfulness, resulting in an overall FAIL — even if every other metric scored perfectly.

### Example — COMP-002 (fraud case)

| Criterion | Model | Score | Weight | Gate |
|-----------|-------|:-----:|:------:|------|
| faithfulness | gemma3:4b | **0.20** | ×3.0 | **< 0.60 → gate FAIL** |
| coverage | gemma3:4b | 1.00 | ×2.0 | ✅ |
| action_orientedness | mistral | 1.00 | ×1.5 | ✅ |
| urgency_tone | mistral | 0.20 | ×2.0 | ✅ |
| entity_preservation | gemma3:4b | 0.80 | ×3.0 | ✅ |

Weighted avg = **0.61** (above fail threshold of 0.55) — but faithfulness gate breached → **FAIL**

Why? SCRIBE wrote "Requesting investigation" in the summary. The customer never said that. `gemma3:4b` flagged it as a hallucinated claim → score 1/5.

---

## Models Used

Three local Ollama models are used, each matched to the appropriate type of reasoning:

| Model | Size | Role |
|-------|------|------|
| `gemma3:4b` | ~3.3 GB | Strict reasoning — used for fraud/compliance critical criteria (faithfulness, entity_preservation) |
| `mistral` | ~4.1 GB | General purpose — primary judge for most criteria |
| `llama3.2` | ~2.0 GB | Lightweight — conciseness, clarity, entity group (combined single call) |

All models run with `temperature=0` for fully reproducible, deterministic scoring.

---

## Per-Case Overrides

Any profile-level setting can be patched for a single case using `"overrides"` in `cases.json`:

```json
"overrides": {
    "entity_preservation": { "enabled": false },
    "faithfulness":        { "judge_model": "gemma3:4b", "weight": 2.5 },
    "coverage":            { "pass_threshold": 0.85, "fail_threshold": 0.65 }
}
```

Overridable fields per criterion:
- `enabled` — `true` or `false` to toggle the metric for this case
- `judge_model` — any Ollama model name (new instance created on demand)
- `weight` — change contribution to the weighted average
- `pass_threshold` — per-criterion pass cutoff
- `fail_threshold` — per-criterion fail cutoff

**The profile itself is never mutated** — overrides are applied per-call only.

---

## Configuration Reference: What You Can Change Without Code

| What to change | Where | Example |
|----------------|-------|---------|
| Add a new complaint | `cases.json` | Add a new JSON block |
| Turn a metric on or off (all cases in a profile) | `evaluation_profiles.yaml` | `enabled: false` |
| Change which model judges a criterion | `evaluation_profiles.yaml` | `judge_model: gemma3:4b` |
| Adjust pass / fail thresholds | `evaluation_profiles.yaml` | `pass_threshold: 0.85` |
| Change a criterion's weight | `evaluation_profiles.yaml` | `weight: 3.0` |
| Add a new routing policy | `policies.yaml` | New `- name:` block |
| Add a new keyword trigger | `policies.yaml` | Add to `keywords:` list |
| Disable a metric for one specific case | `cases.json` overrides | `"enabled": false` |
| Use a different model for one specific case | `cases.json` overrides | `"judge_model": "..."` |
| Add a new evaluation profile | Both YAML files | New profile block + new policy |

---

## Setup and Running

### Prerequisites

```bash
# Install Ollama (https://ollama.com)
ollama pull mistral
ollama pull gemma3:4b
ollama pull llama3.2

pip install -r requirements.txt
```

### Run the evaluation pipeline

```bash
cd complaint_judge
python3 run.py
```

This evaluates all cases in `cases.json`, prints a rich terminal report, and saves a full per-criterion audit to `detailed_report.txt`.

---

## Future Improvements

### Short-term
- **Parallel LLM calls** — run criteria concurrently with `asyncio` for 3–4× speed improvement
- **Prompt tuning** — adjust faithfulness prompt to allow "reasonable inferences" vs strict hallucination detection
- **More keyword triggers** — expand `policies.yaml` with domain-specific keywords

### Medium-term
- **TRACE evaluator** — separate `trace_judge.py` to evaluate TRACE classification quality with its own profiles
- **Score calibration** — compare with human-labelled summaries to tune thresholds
- **Confidence intervals** — run each criterion 2–3 times, flag high-variance criteria for human review

### Long-term
- **REST API** — wrap in FastAPI so any upstream system can POST a complaint and receive a verdict
- **Fine-tuned judge model** — train a small domain-specific model on scored data for faster, cheaper evaluation
- **Feedback loop** — when a human reviewer overrides a verdict, use that signal to retrain thresholds
- **Dashboard** — real-time verdict distribution and score drift monitoring

---

## Design Principles

1. **Zero hardcoding** — every decision lives in YAML. Python only executes.
2. **Strict verdict logic** — one failed dimension on a critical case = FAIL, not hidden in an average.
3. **Per-case flexibility** — override anything at the individual case level without touching shared config.
4. **Keyword-driven routing** — reads actual complaint text, no dependency on upstream metadata tagging.
5. **Reproducible** — `temperature=0` means same input always produces same verdict.
6. **Model-right-sized** — `gemma3:4b` for strict reasoning, `mistral` for general, `llama3.2` for lightweight tasks.
