import google.genai as genai
import json
import re
from typing import Optional
from models.schemas import ExtractionResult, SourceType


EXTRACTION_PROMPT = """You are a senior business analyst AI. Your job is to extract structured business requirements from raw, noisy corporate communications.

Analyze the following {source_type} and extract ALL of the following categories. Be thorough — even implied or partial information counts.

---
{text}
---

Return a single valid JSON object (no markdown, no explanation, just raw JSON) with EXACTLY this structure:

{{
  "functional_requirements": [
    {{
      "id": "FR-001",
      "description": "What the system must DO (specific, actionable)",
      "priority": "P1",
      "confidence": 0.9,
      "source_quote": "exact phrase from text that revealed this"
    }}
  ],
  "non_functional_requirements": [
    {{
      "id": "NFR-001",
      "category": "Performance",
      "description": "The quality constraint",
      "priority": "P1",
      "confidence": 0.8,
      "source_quote": "exact phrase"
    }}
  ],
  "stakeholders": [
    {{
      "name": "Full name or identifier",
      "role": "Their role in the project",
      "decision_authority": "Approver",
      "email": null,
      "mentions": 1
    }}
  ],
  "decisions": [
    {{
      "id": "DEC-001",
      "description": "What was decided",
      "decided_by": "Who decided",
      "date_mentioned": null,
      "status": "Confirmed",
      "source_quote": "exact phrase"
    }}
  ],
  "milestones": [
    {{
      "id": "MS-001",
      "name": "Milestone name",
      "deadline": "date or null",
      "owner": null,
      "status": "Upcoming"
    }}
  ]
}}

RULES:
- priority values must be exactly: "P1", "P2", or "P3"
- decision_authority must be exactly: "Approver", "Influencer", or "Informed"
- status for decisions: "Confirmed", "Pending", or "Disputed"
- status for milestones: "Upcoming", "In Progress", "Overdue", or "Done"
- P1 = Critical/urgent ("must", "required", "ASAP", "blocker", "hard deadline")
- P2 = Important but flexible ("should", "we need", "important")
- P3 = Optional/future ("nice to have", "eventually", "consider")
- confidence: 1.0 = explicitly stated, 0.7 = strongly implied, 0.5 = weakly implied
- Never fabricate — only extract what is actually in the text
- If a category has no items, return an empty array []
- Return ONLY the JSON object, no markdown fences, no explanation
"""


class ExtractionAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def _detect_source_type(self, text: str) -> SourceType:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["from:", "to:", "subject:", "cc:", "sent:"]):
            return SourceType.EMAIL
        if any(kw in text_lower for kw in ["speaker", "transcript", "meeting minutes"]):
            return SourceType.TRANSCRIPT
        return SourceType.CHAT

    def _clean_json(self, raw: str) -> str:
        raw = raw.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return raw.strip()

    def extract(self, text: str, doc_id: str, source_type: Optional[SourceType] = None) -> ExtractionResult:
        if source_type is None:
            source_type = self._detect_source_type(text)

        prompt = EXTRACTION_PROMPT.format(
            source_type=source_type.value,
            text=text[:8000]
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"temperature": 0.1}
        )

        cleaned = self._clean_json(response.text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[ExtractionAgent] JSON parse error for {doc_id}: {e}")
            print(f"Raw response: {cleaned[:300]}")
            data = {
                "functional_requirements": [],
                "non_functional_requirements": [],
                "stakeholders": [],
                "decisions": [],
                "milestones": []
            }

        return ExtractionResult(
            doc_id=doc_id,
            source_type=source_type,
            raw_text_preview=text[:200],
            **data
        )
