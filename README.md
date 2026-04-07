# Complaint LLM-as-a-Judge — Decoupled Profile Edition

A proof of concept for using LLMs to automatically grade banking complaint cases — reducing the manual review workload as complaint volume scales.

The v2 architecture introduces a fully decoupled, YAML-driven evaluation system where **business conditions map to evaluation profiles**, which map to specific criteria, judge models, weights, and thresholds — all configurable without touching Python code.

---

## Architecture

```
Business Condition (complaint type, severity, department, risk level)
         │
         ▼  profile_routing.yaml  (first-match routing rules)
Evaluation Profile  (standard / stringent / fraud_focused / loan_focused)
         │
         ▼  evaluation_profiles.yaml
  Per-Criterion Config:
    ├── enabled?         (true/false — skip entirely if false)
    ├── judge_model      (which Ollama model evaluates this criterion)
    ├── weight           (influence on weighted average score)
    ├── pass_threshold   (per-criterion override, falls back to profile-level)
    └── fail_threshold   (per-criterion override, falls back to profile-level)
         │
         ▼
  Verdict (Option B — strict):
    PASS         → weighted_avg ≥ profile.pass_threshold
                   AND no single criterion below its own fail_threshold
    FAIL         → weighted_avg < profile.fail_threshold
                   OR any individual criterion below its fail_threshold
    NEEDS_REVIEW → everything else
```

---

## Profiles

| Profile | Description | Pass ≥ | Fail < | Models Used |
|---|---|---|---|---|
| `standard` | General/low-risk complaints | 0.75 | 0.50 | mistral, llama3.2 |
| `stringent` | High-risk / compliance | 0.80 | 0.60 | gemma3:4b, mistral, llama3.2 |
| `fraud_focused` | Fraud / identity theft | 0.78 | 0.55 | gemma3:4b, mistral, llama3.2 |
| `loan_focused` | Loan / mortgage complaints | 0.75 | 0.50 | mistral, llama3.2 |

### Why different models per profile?

| Model | Role | Used for |
|---|---|---|
| `gemma3:4b` | Strict reasoning judge | Faithfulness/entity checks in fraud & compliance |
| `mistral` | General fast judge | Primary judge for most criteria |
| `llama3.2` | Lightweight judge | Combined metrics (conciseness, clarity, entity) |

> **Why a separate model for each criterion?**
> Just as you don't let a student mark their own exam, different LLMs have different reasoning styles. Using a stricter model (gemma3:4b) for fraud entity checks — while keeping a fast model (llama3.2) for lightweight metrics — gives both quality and speed.

---

## Routing Rules

Rules in `config/profile_routing.yaml` are evaluated **top → bottom**. First match wins.

| Condition | Profile |
|---|---|
| Category = `Fraud/Identity Theft` | `fraud_focused` |
| Category starts with `Fraud/` | `fraud_focused` |
| `risk_level=high` AND `complaint_type=fraud` | `fraud_focused` |
| `severity=high` AND `department=compliance` | `stringent` |
| `risk_level=high` (any type) | `stringent` |
| `severity=high` (any dept) | `stringent` |
| Category starts with `Loan/` | `loan_focused` |
| `complaint_type=loan` | `loan_focused` |
| `severity=medium` AND `department=compliance` | `stringent` |
| *(fallthrough)* | `standard` |

---

## Criteria Reference

| Criterion | What it checks | Standard | Stringent | Fraud | Loan |
|---|---|:---:|:---:|:---:|:---:|
| `faithfulness` | Did the summary hallucinate anything? | ✅ ×1.5 | ✅ ×2.0 | ✅ ×3.0 | ✅ ×1.5 |
| `coverage` | Did it capture problem, ask, urgency? | ✅ ×1.0 | ✅ ×2.0 | ✅ ×2.0 | ✅ ×2.0 |
| `action_orientedness` | Can an agent immediately act on it? | ✅ ×1.0 | ✅ ×1.5 | ✅ ×1.5 | ✅ ×2.0 |
| `conciseness` | Is it appropriately compressed? | ✅ ×0.75 | ✅ ×1.0 | ❌ off | ✅ ×0.75 |
| `clarity` | Is it understandable in 5 seconds? | ✅ ×0.75 | ✅ ×1.0 | ✅ ×0.75 | ✅ ×1.0 |
| `urgency_tone` | Does tone match original severity? | ❌ off | ✅ ×1.5 | ✅ ×2.0 | ✅ ×1.5 |
| `entity_preservation` | Are amounts/dates/IDs retained? | ✅ ×1.0 | ✅ ×2.0 | ✅ ×3.0 | ✅ ×1.5 |
| `trace_classification` | Is the complaint category correct? | ✅ ×1.5 | ✅ ×2.0 | ✅ ×3.0 | ✅ ×1.5 |

> **Weights** (×N) directly influence the weighted average. A `faithfulness` score of 0.4 in `fraud_focused` has 3× the drag compared to the same score in `standard`.

---

## File Structure

```
complaint_judge/
├── config/
│   ├── evaluation_profiles.yaml   # Profile definitions (criteria, models, thresholds)
│   └── profile_routing.yaml       # Business condition → profile routing rules
├── judge.py                        # Profile-driven LLM Judge (zero hardcoded values)
├── profile_loader.py              # YAML loader + profile resolver
├── cases.py                       # Test cases with severity/department/risk_level metadata
├── run.py                         # Orchestrates grading + prints rich output
├── test_profiles.py               # Config & routing unit tests (no LLM calls)
├── requirements.txt               # langchain-ollama, rich, PyYAML
└── detailed_report.txt            # Generated audit log after run.py
```

---

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally.

Pull the required models:
```bash
ollama pull mistral          # primary general judge
ollama pull gemma3:4b        # strict reasoning judge (fraud/compliance)
ollama pull llama3.2         # lightweight judge (conciseness, clarity)
```

Install Python dependencies:
```bash
pip3 install -r requirements.txt
```

---

## Running

### Run the full evaluation pipeline:
```bash
python3 run.py
```

### Run config + routing tests (no LLM calls, fast):
```bash
python3 test_profiles.py
```

---

## Customisation (Zero Code Required)

All of the following can be changed by editing YAML files only — no Python changes needed:

### Turn a metric on or off
```yaml
# In config/evaluation_profiles.yaml
urgency_tone:
  enabled: false   # ← flip to true to activate
```

### Swap the judge model for a criterion
```yaml
faithfulness:
  judge_model: gemma3:4b   # ← change to any ollama model
```

### Adjust scoring thresholds
```yaml
# Profile-level
pass_threshold: 0.80
fail_threshold: 0.60

# Per-criterion override
faithfulness:
  pass_threshold: 0.90    # stricter than profile default
  fail_threshold: 0.70
```

### Add a new routing rule
```yaml
# In config/profile_routing.yaml
routing_rules:
  - name: "VIP customer complaints"
    conditions:
      customer_tier: vip
    profile: stringent
```

### Add a new profile
Copy an existing profile block in `evaluation_profiles.yaml`, name it, adjust criteria, and add a routing rule to point to it. No Python required.

---

## Scoring Details

### Weighted Score

```
weighted_avg = Σ(score_i × weight_i) / Σ(weight_i)
               for all enabled criteria
```

### Verdict Logic (Option B — Strict)

```
PASS         → weighted_avg ≥ pass_threshold
               AND no criterion score < its fail_threshold

FAIL         → weighted_avg < fail_threshold
               OR any criterion score < its own fail_threshold

NEEDS_REVIEW → everything else
```

This means a single badly-scored criterion (e.g., `faithfulness = 0.2` in `fraud_focused`) will trigger `FAIL` regardless of the weighted average — matching real-world compliance requirements.

### Non-LLM Deterministic Metrics (always run)

| Metric | Formula | Acceptable Range |
|---|---|---|
| `compression_ratio` | `len(summary) / len(complaint)` | 0.10 – 0.40 |
| `entity_recall` | regex-extracted entities found in summary | — |

---

## Test Cases

| Case | Complaint Type | Severity | Department | → Profile | Note |
|---|---|---|---|---|---|
| COMP-001 | Billing | low | billing | `standard` | Good summary |
| COMP-002 | Fraud/Transaction | high | fraud | `fraud_focused` | Good summary |
| COMP-003 | Loan/Mortgage | medium | loans | `loan_focused` | **Weak summary** — judge should catch |
| COMP-004 | Fraud/Identity | high | fraud | `fraud_focused` | **Wrong category** — judge should catch |
| COMP-005 | Account Access | medium | customer_service | `standard` | Good summary |
| COMP-006 | Billing/Compliance | high | compliance | `stringent` | Regulatory risk |
| COMP-007 | Customer Service | low | customer_service | `standard` | Good summary |
