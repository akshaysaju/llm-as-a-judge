# Complaint LLM-as-a-Judge POC

A proof of concept for using an LLM to automatically grade banking complaint cases — reducing the manual review workload as complaint volume scales.

---

## Problem

**SCRIBE** summarizes incoming customer complaints in real-time. A sample of those cases is sent to a manual review queue where humans check:
- Is the summary accurate?
- Is the complaint category correct?
- Did the agent actually resolve the issue?

As adoption scales in 2026, the manual review queue grows faster than the team can handle it.

---

## Solution

An **LLM Judge** grades each case automatically and routes it:

```
PASS          → auto-approved, removed from manual queue
NEEDS_REVIEW  → sent to human reviewer
FAIL          → escalated to senior reviewer
```

This reduces manual workload — only borderline and failed cases reach a human.

---

## How it works

```
Complaint Case (already processed by SCRIBE + TRACE + Agent)
        │
        ▼
   ┌──────────┐
   │  JUDGE   │  reads: raw complaint + SCRIBE summary + TRACE category + agent resolution
   │ (mistral)│
   └────┬─────┘
        │  grades 3 things independently
        ├── SCRIBE Grade      → is the summary accurate and complete?
        ├── TRACE Grade       → is the complaint category correct?
        └── Resolution Grade  → did the agent actually resolve the issue?
                │
        ┌───────┴────────┐
        │  Routing Logic  │
        │  all ≥ 0.75    →  PASS
        │  any  < 0.50   →  FAIL
        │  otherwise     →  NEEDS_REVIEW
        └────────────────┘
```

### Why a different model as Judge?

SCRIBE and TRACE use `llama3.2`. The Judge uses `mistral` — an independent model that has not seen these outputs before, so its grading is unbiased. Same reason you don't let a student mark their own exam.

---

## Files

| File | Purpose |
|---|---|
| `cases.py` | Sample complaint cases with SCRIBE summaries, TRACE categories, and agent resolutions |
| `judge.py` | The LLM Judge — grades SCRIBE, TRACE, and resolution independently |
| `run.py` | Orchestrates the workflow and shows HITL routing results |

---

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally.

Pull the required models:
```bash
ollama pull llama3.2
ollama pull mistral
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run:
```bash
python run.py
```

---

## Example Output

```
COMP-002  PASS          SCRIBE:1.00  TRACE:1.00  RESOLUTION:0.95  → auto-approved
COMP-005  PASS          SCRIBE:0.95  TRACE:1.00  RESOLUTION:0.90  → auto-approved
COMP-003  NEEDS_REVIEW  SCRIBE:0.70  TRACE:1.00  RESOLUTION:0.65  → human queue
COMP-004  NEEDS_REVIEW  SCRIBE:1.00  TRACE:1.00  RESOLUTION:0.60  → human queue
COMP-001  NEEDS_REVIEW  SCRIBE:1.00  TRACE:0.50  RESOLUTION:0.95  → human queue

Manual review workload reduced by 40% (2 of 5 cases handled by LLM Judge)
```

---

## Routing Thresholds

| Threshold | Verdict | Meaning |
|---|---|---|
| All scores ≥ 0.75 | `PASS` | Case is good — auto-approved |
| Any score < 0.50 | `FAIL` | Something seriously wrong — escalate |
| Otherwise | `NEEDS_REVIEW` | Human should review |

Thresholds are configurable in `judge.py`.
