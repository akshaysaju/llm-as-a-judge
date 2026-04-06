"""
judge.py — LLM as a Judge
──────────────────────────
This is the core of the system.

The Judge receives a complaint case that has already been processed by:
  - SCRIBE  (produced a summary)
  - TRACE   (produced a category)
  - Agent   (produced a resolution)

It grades three things independently:

  1. SCRIBE Grade   — Is the summary accurate and complete?
  2. TRACE Grade    — Is the complaint category correct?
  3. Resolution Grade — Did the agent actually resolve the issue?

Each score is 0.0 – 1.0.
The final verdict drives the human-in-the-loop routing:

  ALL scores ≥ 0.75  →  PASS          (auto-approved, removed from manual queue)
  ANY score  < 0.50  →  FAIL          (escalate to senior reviewer)
  otherwise          →  NEEDS_REVIEW  (goes to standard human review queue)

Why a different model as Judge?
  SCRIBE and TRACE use llama3.2. The Judge uses mistral — an independent
  model that has not seen these outputs before, so its grading is unbiased.
"""

import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

JUDGE_MODEL = "mistral"

VALID_CATEGORIES = [
    "Billing/Fees Dispute",
    "Fraud/Unauthorised Transaction",
    "Fraud/Identity Theft",
    "Loan/Mortgage Complaint",
    "Account Access Issue",
    "Customer Service Complaint",
    "Other",
]

# ── Prompts ───────────────────────────────────────────────────────────────────

SCRIBE_GRADE_PROMPT = ChatPromptTemplate.from_template("""
You are grading a complaint summary produced by an automated summarizer (SCRIBE).

Original Complaint:
{complaint}

SCRIBE Summary:
{summary}

Grade the summary on:
- accuracy: are all facts in the summary correct and present in the original?
- completeness: does the summary capture the core issue and what the customer wants?

Reply with JSON only: {{"accuracy": 0.0-1.0, "completeness": 0.0-1.0, "note": "one sentence"}}
""")

TRACE_GRADE_PROMPT = ChatPromptTemplate.from_template("""
You are grading a complaint classification produced by an automated classifier (TRACE).

Original Complaint:
{complaint}

TRACE assigned category: {category}
Valid categories: {valid_categories}

Grade the classification:
- 1.0 if the category is correct
- 0.5 if it is a reasonable alternative but not the best fit
- 0.0 if it is clearly wrong

Reply with JSON only: {{"score": 0.0-1.0, "note": "one sentence"}}
""")

RESOLUTION_GRADE_PROMPT = ChatPromptTemplate.from_template("""
You are grading an agent's resolution of a customer complaint.

Original Complaint:
{complaint}

Agent Resolution:
{resolution}

Grade the resolution on:
- addressed: did the agent actually tackle the customer's specific problem?
- quality: is the action taken appropriate and sufficient?

Reply with JSON only: {{"addressed": 0.0-1.0, "quality": 0.0-1.0, "note": "one sentence"}}
""")

# ── Judge ─────────────────────────────────────────────────────────────────────

class Judge:
    def __init__(self):
        llm    = ChatOllama(model=JUDGE_MODEL, temperature=0)
        parser = StrOutputParser()
        self.scribe_chain     = SCRIBE_GRADE_PROMPT     | llm | parser
        self.trace_chain      = TRACE_GRADE_PROMPT      | llm | parser
        self.resolution_chain = RESOLUTION_GRADE_PROMPT | llm | parser

    def _parse(self, text):
        try:
            m = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(m.group()) if m else {}
        except Exception:
            return {}

    def grade(self, case: dict) -> dict:
        """Grade a single complaint case. Returns scores and routing verdict."""

        # Grade SCRIBE summary
        s = self._parse(self.scribe_chain.invoke({
            "complaint": case["complaint"],
            "summary":   case["scribe_summary"],
        }))
        scribe_accuracy    = round(float(s.get("accuracy", 0)), 2)
        scribe_completeness= round(float(s.get("completeness", 0)), 2)
        scribe_score       = round((scribe_accuracy + scribe_completeness) / 2, 2)
        scribe_note        = s.get("note", "")

        # Grade TRACE classification
        t = self._parse(self.trace_chain.invoke({
            "complaint":        case["complaint"],
            "category":         case["trace_category"],
            "valid_categories": ", ".join(VALID_CATEGORIES),
        }))
        trace_score = round(float(t.get("score", 0)), 2)
        trace_note  = t.get("note", "")

        # Grade agent resolution
        r = self._parse(self.resolution_chain.invoke({
            "complaint":  case["complaint"],
            "resolution": case["agent_resolution"],
        }))
        resolution_addressed = round(float(r.get("addressed", 0)), 2)
        resolution_quality   = round(float(r.get("quality", 0)), 2)
        resolution_score     = round((resolution_addressed + resolution_quality) / 2, 2)
        resolution_note      = r.get("note", "")

        # Routing verdict
        scores  = [scribe_score, trace_score, resolution_score]
        avg     = round(sum(scores) / len(scores), 2)
        verdict = self._verdict(scores)

        return {
            "id": case["id"],
            # Individual sub-scores per dimension
            "individual_scores": {
                "scribe_accuracy":        scribe_accuracy,
                "scribe_completeness":    scribe_completeness,
                "trace_classification":   trace_score,
                "resolution_addressed":   resolution_addressed,
                "resolution_quality":     resolution_quality,
            },
            # Aggregated scores (average of sub-scores per component)
            "scores": {
                "scribe_summary":   scribe_score,
                "trace_category":   trace_score,
                "agent_resolution": resolution_score,
            },
            "notes": {
                "scribe":     scribe_note,
                "trace":      trace_note,
                "resolution": resolution_note,
            },
            "avg":     avg,
            "verdict": verdict,
        }

    def _verdict(self, scores):
        if min(scores) < 0.50:  return "FAIL"
        if min(scores) >= 0.75: return "PASS"
        return "NEEDS_REVIEW"
