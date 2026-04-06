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
   │  JUDGE   │  reads: raw complaint + SCRIBE summary + TRACE category
   │ (mistral)│
   └────┬─────┘
        │  grades 2 things independently
        ├── SCRIBE Grade      → is the summary accurate, faithful, and action-oriented?
        └── TRACE Grade       → is the complaint category correct?
                │
        ┌───────┴────────┐
        │  Routing Logic  │
        │  min ≥ 0.75    →  PASS
        │  min < 0.50    →  FAIL
        │  otherwise     →  NEEDS_REVIEW
        └────────────────┘
```

### Why a different model as Judge?

SCRIBE and TRACE use `llama3.2`. The Judge uses `mistral` — an independent model that has not seen these outputs before, so its grading is unbiased. Same reason you don't let a student mark their own exam.

---

## Files

| File | Purpose |
|---|---|
| `cases.py` | Sample complaint cases with SCRIBE summaries and TRACE categories |
| `judge.py` | The LLM Judge — grades SCRIBE and TRACE output independently |
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

Run the pipeline:
```bash
python3 run.py
```

Run the consistency check test:
```bash
python3 test_consistency.py
```

---

## Scoring Details

The system calculates scores using a combination of LLM evaluation, normalisation, and deterministic logic:

1. **Raw LLM Scoring (1 to 5):** The LLM evaluates the complaint and summary against specific rubrics via Chain-of-Thought, assigning an integer score. A `1` represents failure (e.g., hallucinated facts), while a `5` represents perfect execution.
2. **Normalisation (0.0 to 1.0):** The `1-5` integer is mathematically divided by 5 to create a strict percentage score (e.g., an LLM score of 3 becomes `0.60`).
3. **SCRIBE Average:** The final SCRIBE score is an average of 7 distinct LLM dimensions. 
4. **Deterministic Metrics:** The script independently assesses the summaries using pure Python:
   - *Compression Ratio:* `len(summary) / len(complaint)`. Summaries manually checked against a strict `0.10 - 0.40` acceptable range. 
   - *Entity Recall:* Uses Regex to strictly check what percentage of extracted entities (e.g., `$35`, `March 28th`) survived the summarisation.

### Metric Definitions (What and Why)
The Judge evaluates the upstream systems across these specific dimensions:

#### SCRIBE Metrics (Summarisation Quality)
*   **Faithfulness:** *Did the summary make things up?* Important to ensure the AI did not hallucinate completely new facts not present in the original complaint.
*   **Coverage:** *Did it miss the main point?* Verifies the summary successfully captured the core problem, the impact on the customer, and the customer's actual ask.
*   **Action-Orientedness:** *Can an agent act on this?* A support agent should be able to instantly read the summary and know the exact next step without digging into the original complaint.
*   **Conciseness:** *Is it bloated?* The goal of SCRIBE is compression. If it's too vague or too long, it fails its purpose.
*   **Clarity:** *Is it an easy read?* Ensures the grammar and structure can be comprehended by a human agent in under 5 seconds.
*   **Urgency Tone:** *Is the customer screaming?* If the original text is highly severe or angry, the summary must retain that tone so human queues can prioritise the ticket appropriately.
*   **Entity Preservation:** *Did we lose the data?* Crucial check to ensure IDs, dates (e.g., `March 28th`), and monetary amounts (e.g., `$4,200`) weren't stripped out by the summariser.

#### TRACE Metrics (Classification Quality)
*   **Classification:** *Did it go to the right department?* Ensures the ticket was correctly assigned to a category (e.g., *Fraud* vs *Billing Dispute*). If misclassified, it creates immense friction in the support pipeline.

### Prompt Example
The judge asks the LLM to provide step-by-step reasoning *before* scoring to prevent score drift.

**Example: Faithfulness Prompt**
```text
You are evaluating a complaint summary for faithfulness.

Step 1 — List every factual claim in the summary.
Step 2 — For each claim, check if it is present in the original complaint.
Step 3 — Score using this rubric:
  1 = Summary contains facts not present in the complaint (hallucination)
  2 = Summary has significant paraphrasing that distorts meaning
  3 = Summary is mostly faithful, minor wording differences only
  4 = All claims traceable but one small detail is off
  5 = Every claim in the summary can be traced directly to the complaint

Original Complaint:
{complaint}

SCRIBE Summary:
{summary}

Reply with JSON only. Example: {{"reasoning": "your step-by-step analysis", "score": 4}}
```

---

## Example Output

```text
  Case       Faith   Cover   Action   Concise   Clarity   Urgency   Entity   TRACE   Ratio           Verdict       
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  COMP-001    0.60    0.60     1.00      0.60      1.00      0.60     1.00    1.00   0.6 too_long    PASS          
  COMP-002    0.60    0.60     1.00      0.60      1.00      0.60     1.00    1.00   0.64 too_long   PASS          
  COMP-003    0.80    0.40     0.80      0.20      0.60      0.60     0.40    1.00   0.26 ok         NEEDS_REVIEW  
  COMP-004    0.80    0.40     1.00      0.60      1.00      0.60     1.00    0.00   0.6 too_long    FAIL          
  COMP-005    0.60    0.40     1.00      0.60      1.00      0.80     1.00    1.00   0.93 too_long   PASS          

Routing
  PASS         3/5 — auto-approved
  NEEDS_REVIEW 1/5 — human queue
  FAIL         1/5 — escalated

  Workload reduced by 60%
```

---

## Routing Thresholds

| Threshold | Verdict | Meaning |
|---|---|---|
| All scores ≥ 0.75 | `PASS` | Case is good — auto-approved |
| Any score < 0.50 | `FAIL` | Something seriously wrong — escalate |
| Otherwise | `NEEDS_REVIEW` | Human should review |

Thresholds are configurable in `judge.py`.
