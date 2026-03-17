import google.genai as genai
import json
import re
from typing import List
from models.schemas import ExtractionResult, Conflict


CONFLICT_PROMPT = """You are a requirements analyst checking for contradictions across multiple business documents.

Below are extracted requirements and milestones from {n_docs} different documents. Identify any conflicts, contradictions, or ambiguities.

EXTRACTED DATA:
{data}

Return a JSON array of conflicts (empty array [] if none found):

[
  {{
    "conflict_id": "CONF-001",
    "type": "Deadline conflict | Requirement contradiction | Ownership dispute | Scope disagreement",
    "description": "Clear explanation of what conflicts and why it matters",
    "source_a": "First statement (with doc_id reference)",
    "source_b": "Contradicting statement (with doc_id reference)",
    "severity": "High | Medium | Low"
  }}
]

RULES:
- High severity: directly contradicting requirements or deadlines that differ by more than a week
- Medium severity: same feature described differently, unclear ownership, ambiguous priority
- Low severity: minor wording differences, implied vs explicit
- Only flag real conflicts — don't hallucinate
- Return raw JSON only, no markdown
"""


class ConflictDetectorAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def _clean_json(self, raw: str) -> str:
        raw = raw.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return raw.strip()

    def _summarize_for_conflict(self, results: List[ExtractionResult]) -> str:
        """Build a compact summary of all extractions for the conflict prompt"""
        lines = []
        for r in results:
            lines.append(f"\n=== Document: {r.doc_id} ({r.source_type}) ===")

            if r.milestones:
                lines.append("MILESTONES:")
                for m in r.milestones:
                    lines.append(f"  - {m.name}: deadline={m.deadline}, owner={m.owner}")

            if r.functional_requirements:
                lines.append("FUNCTIONAL REQUIREMENTS:")
                for fr in r.functional_requirements:
                    lines.append(f"  - [{fr.priority}] {fr.description}")

            if r.decisions:
                lines.append("DECISIONS:")
                for d in r.decisions:
                    lines.append(f"  - {d.description} (by {d.decided_by}, status={d.status})")

            if r.stakeholders:
                lines.append("STAKEHOLDERS:")
                for s in r.stakeholders:
                    lines.append(f"  - {s.name} ({s.role}): {s.decision_authority}")

        return "\n".join(lines)

    def detect(self, results: List[ExtractionResult]) -> List[Conflict]:
        """Detect conflicts across multiple extraction results"""
        if len(results) < 2:
            return []  # Need at least 2 docs to find conflicts

        summary = self._summarize_for_conflict(results)
        prompt = CONFLICT_PROMPT.format(n_docs=len(results), data=summary)

        response = self.client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config={"temperature": 0.1}
        )
        cleaned = self._clean_json(response.text)

        try:
            raw_conflicts = json.loads(cleaned)
            if not isinstance(raw_conflicts, list):
                raw_conflicts = []
        except json.JSONDecodeError:
            raw_conflicts = []

        conflicts = []
        for i, c in enumerate(raw_conflicts):
            try:
                conflicts.append(Conflict(**c))
            except Exception as e:
                print(f"[ConflictDetector] Skipping malformed conflict {i}: {e}")

        return conflicts
