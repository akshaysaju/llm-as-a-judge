"""
judge.py — LLM as a Judge
──────────────────────────
Grades SCRIBE (summary) and TRACE (classification) outputs.

SCRIBE evaluation — 4 calls total:
  Call 1: Faithfulness        — most critical, separate call
  Call 2: Coverage            — most critical, separate call
  Call 3: Action-Orientedness — most critical, separate call
  Call 4: Conciseness + Clarity + Urgency + Entity Preservation — combined

TRACE evaluation — 1 call:
  Call 5: Classification accuracy

Each call uses:
  - Chain-of-thought before scoring (reason first, then score)
  - Rubric anchors so the model has no room to drift
  - Score 1–5, normalised to 0.0–1.0
  - Temperature 0 for consistency

Routing verdict:
  ALL scores >= 0.75  →  PASS         (auto-approved)
  ANY score  <  0.50  →  FAIL         (escalate)
  otherwise           →  NEEDS_REVIEW (human queue)
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

# Call 1 — Faithfulness
FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template("""
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
Use an actual integer for score.
""")

# Call 2 — Coverage
COVERAGE_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating a complaint summary for coverage.

Step 1 — Identify in the complaint: (a) core problem, (b) impact on customer, (c) urgency signals, (d) what the customer is asking for.
Step 2 — Check which of these appear in the summary.
Step 3 — Score using this rubric:
  1 = Core problem is missing from the summary
  2 = Core problem present but customer ask or impact is missing
  3 = Core problem and ask present but urgency or impact is missing
  4 = All key parts present, one minor detail missing
  5 = Summary fully captures problem, impact, urgency signals, and customer ask

Original Complaint:
{complaint}

SCRIBE Summary:
{summary}

Reply with JSON only. Example: {{"reasoning": "your step-by-step analysis", "score": 3}}
Use an actual integer for score.
""")

# Call 3 — Action-Orientedness
ACTION_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating whether a complaint summary is action-oriented.

Step 1 — Identify what specific action the customer wants (refund, fix, callback, investigation, etc.).
Step 2 — Check if a support agent reading only the summary would immediately know what to do next.
Step 3 — Score using this rubric:
  1 = Summary gives no indication of what action is needed
  2 = Action is vaguely implied but not clear
  3 = Action is mentioned but lacks specifics (amount, date, account)
  4 = Action is clear with most relevant details
  5 = Agent knows exactly what to do from the summary alone — action, amount, date, account all present

Original Complaint:
{complaint}

SCRIBE Summary:
{summary}

Reply with JSON only. Example: {{"reasoning": "your step-by-step analysis", "score": 5}}
Use an actual integer for score.
""")

# Call 4 — Combined (simpler dimensions)
COMBINED_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating a complaint summary on four dimensions. Score each 1–5.

Rubrics:
  Conciseness       — 1: too long or too vague  3: acceptable length  5: perfectly compressed
  Clarity           — 1: confusing or ambiguous  3: mostly clear  5: agent understands in 5 seconds
  Urgency Tone      — 1: angry complaint summarised as calm  3: partial  5: tone and severity match original
  Entity Preservation — 1: amounts/dates/IDs missing  3: some preserved  5: all key entities retained

Original Complaint:
{complaint}

SCRIBE Summary:
{summary}

Reply with JSON only:
{{
  "conciseness":          {{"score": 1-5, "reason": "one sentence"}},
  "clarity":              {{"score": 1-5, "reason": "one sentence"}},
  "urgency_tone":         {{"score": 1-5, "reason": "one sentence"}},
  "entity_preservation":  {{"score": 1-5, "reason": "one sentence"}}
}}
""")

# Call 5 — TRACE classification
TRACE_PROMPT = ChatPromptTemplate.from_template("""
You are grading a complaint classification produced by TRACE.

Step 1 — Identify the primary nature of the complaint.
Step 2 — Check if the assigned category matches.
Step 3 — Assign a score:
  Use 1.0 if the category is correct.
  Use 0.5 if it is a reasonable alternative but not the best fit.
  Use 0.0 if it is clearly wrong.

Original Complaint:
{complaint}

TRACE assigned: {category}
Valid categories: {valid_categories}

Reply with JSON only. Example format: {{"reasoning": "one sentence", "score": 1.0}}
Use an actual number for score, not a range.
""")


# ── Non-LLM metrics (fast, deterministic) ─────────────────────────────────────

import re as _re

def compression_ratio(complaint: str, summary: str) -> float:
    """Length of summary / length of complaint. Ideal range: 0.10 – 0.35."""
    c = len(complaint.split())
    s = len(summary.split())
    return round(s / c, 2) if c > 0 else 0.0

def entity_recall(complaint: str, summary: str) -> float:
    """
    Extracts key entities (amounts, dates, IDs) from the complaint
    and checks how many appear in the summary.
    """
    # Extract amounts ($35, $4,200, $249.99), dates (March 28, April 1st),
    # and numeric IDs (4 days, 6 weeks, 11 years, 3 times)
    pattern = r'\$[\d,\.]+|\b\d+[\s-]?(days?|weeks?|months?|years?|times?)\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+\w*\b|\bapril\s+\d+\b'
    entities = _re.findall(pattern, complaint.lower())
    entities = [e if isinstance(e, str) else ' '.join(e).strip() for e in entities]
    entities = [e for e in entities if e.strip()]

    if not entities:
        return 1.0  # no entities to check

    found = sum(1 for e in entities if e.lower() in summary.lower())
    return round(found / len(entities), 2)

def compression_flag(ratio: float) -> str:
    """Flag if compression ratio is outside acceptable range."""
    if ratio < 0.10: return "too_short"
    if ratio > 0.40: return "too_long"
    return "ok"


# ── Judge ─────────────────────────────────────────────────────────────────────

class Judge:
    def __init__(self):
        llm    = ChatOllama(model=JUDGE_MODEL, temperature=0)
        parser = StrOutputParser()
        self.faithfulness_chain = FAITHFULNESS_PROMPT | llm | parser
        self.coverage_chain     = COVERAGE_PROMPT     | llm | parser
        self.action_chain       = ACTION_PROMPT       | llm | parser
        self.combined_chain     = COMBINED_PROMPT     | llm | parser
        self.trace_chain        = TRACE_PROMPT        | llm | parser

    def _parse(self, text: str) -> dict:
        try:
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                # Strip control characters that break JSON parsing
                clean = re.sub(r'[\x00-\x1f\x7f]', ' ', m.group())
                return json.loads(clean)
        except Exception:
            pass
        return {}

    def _norm(self, score, out_of=5) -> float:
        """Normalise 1–5 scale to 0.0–1.0."""
        return round(float(score) / out_of, 2)

    def grade(self, case: dict) -> dict:
        complaint = case["complaint"]
        summary   = case["scribe_summary"]

        # ── Call 1: Faithfulness ──────────────────────────────────────────────
        raw_f = self.faithfulness_chain.invoke({"complaint": complaint, "summary": summary})
        f = self._parse(raw_f)
        faithfulness = self._norm(f.get("score", 1))
        f_reason     = f.get("reasoning", "")

        # ── Call 2: Coverage ──────────────────────────────────────────────────
        raw_c = self.coverage_chain.invoke({"complaint": complaint, "summary": summary})
        c = self._parse(raw_c)
        coverage  = self._norm(c.get("score", 1))
        c_reason  = c.get("reasoning", "")

        # ── Call 3: Action-Orientedness ───────────────────────────────────────
        raw_a = self.action_chain.invoke({"complaint": complaint, "summary": summary})
        a = self._parse(raw_a)
        action    = self._norm(a.get("score", 1))
        a_reason  = a.get("reasoning", "")

        # ── Call 4: Combined (conciseness, clarity, urgency, entities) ────────
        raw_cmb = self.combined_chain.invoke({"complaint": complaint, "summary": summary})
        cmb = self._parse(raw_cmb)
        conciseness  = self._norm(cmb.get("conciseness",  {}).get("score", 1))
        clarity      = self._norm(cmb.get("clarity",      {}).get("score", 1))
        urgency      = self._norm(cmb.get("urgency_tone", {}).get("score", 1))
        entities     = self._norm(cmb.get("entity_preservation", {}).get("score", 1))

        # ── Call 5: TRACE classification ──────────────────────────────────────
        raw_t = self.trace_chain.invoke({
            "complaint":        complaint,
            "category":         case["trace_category"],
            "valid_categories": ", ".join(VALID_CATEGORIES),
        })
        t = self._parse(raw_t)
        trace_score  = round(float(t.get("score", 0)), 2)
        t_reason     = t.get("reasoning", "")

        # ── Non-LLM metrics ───────────────────────────────────────────────────
        ratio  = compression_ratio(complaint, summary)
        recall = entity_recall(complaint, summary)
        flag   = compression_flag(ratio)

        # ── Scoring ───────────────────────────────────────────────────────────
        scribe_scores = {
            "faithfulness":        faithfulness,
            "coverage":            coverage,
            "action_orientedness": action,
            "conciseness":         conciseness,
            "clarity":             clarity,
            "urgency_tone":        urgency,
            "entity_preservation": entities,
        }
        scribe_avg = round(sum(scribe_scores.values()) / len(scribe_scores), 2)

        verdict = self._verdict([scribe_avg, trace_score])

        return {
            "id":      case["id"],
            "scribe":  scribe_scores,
            "scribe_avg": scribe_avg,
            "scribe_reasons": {
                "faithfulness":        f_reason,
                "coverage":            c_reason,
                "action_orientedness": a_reason,
                "conciseness":  cmb.get("conciseness",  {}).get("reason", ""),
                "clarity":      cmb.get("clarity",      {}).get("reason", ""),
                "urgency_tone": cmb.get("urgency_tone", {}).get("reason", ""),
                "entity_preservation": cmb.get("entity_preservation", {}).get("reason", ""),
            },
            "trace":  {"score": trace_score, "reason": t_reason},
            "non_llm": {
                "compression_ratio": ratio,
                "compression_flag":  flag,
                "entity_recall":     recall,
            },
            "verdict": verdict,
            "raw": {
                "faithfulness":  raw_f,
                "coverage":      raw_c,
                "action":        raw_a,
                "combined":      raw_cmb,
                "trace":         raw_t
            }
        }

    def _verdict(self, scores: list) -> str:
        if min(scores) < 0.50:   return "FAIL"
        if min(scores) >= 0.75:  return "PASS"
        return "NEEDS_REVIEW"
