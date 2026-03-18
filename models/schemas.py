from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class Priority(str, Enum):
    P1 = "P1"  # Critical / Must Have
    P2 = "P2"  # Important / Should Have
    P3 = "P3"  # Nice to Have


class SourceType(str, Enum):
    EMAIL = "email"
    TRANSCRIPT = "transcript"
    CHAT = "chat"


class FunctionalRequirement(BaseModel):
    id: str = Field(description="Unique ID e.g. FR-001")
    description: str = Field(description="What the system must DO")
    priority: Priority
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence 0-1")
    source_quote: str = Field(description="Exact phrase from source that triggered this")


class NonFunctionalRequirement(BaseModel):
    id: str = Field(description="Unique ID e.g. NFR-001")
    category: str = Field(description="e.g. Performance, Security, Scalability, Availability")
    description: str
    priority: Priority
    confidence: float = Field(ge=0.0, le=1.0)
    source_quote: str


class Stakeholder(BaseModel):
    name: str
    role: str = Field(description="e.g. Product Manager, Engineer, Client")
    decision_authority: str = Field(description="Approver / Influencer / Informed")
    email: Optional[str] = None
    mentions: int = Field(default=1, description="How many times they appear across docs")


class Decision(BaseModel):
    id: str = Field(description="e.g. DEC-001")
    description: str = Field(description="What was decided")
    decided_by: Optional[str] = Field(default="Unknown", description="Who decided")
   # decided_by: str = Field(description="Who made the decision")
    date_mentioned: Optional[str] = None
    status: str = Field(description="Confirmed / Pending / Disputed")
    source_quote: str


class Milestone(BaseModel):
    id: str = Field(description="e.g. MS-001")
    name: str
    deadline: Optional[str] = Field(None, description="ISO date or natural language")
    owner: Optional[str] = None
    status: str = Field(description="Upcoming / In Progress / Overdue / Done")


class Conflict(BaseModel):
    conflict_id: str
    type: str = Field(description="Deadline conflict / Requirement contradiction / Ownership dispute")
    description: str
    source_a: str = Field(description="First conflicting statement")
    source_b: str = Field(description="Contradicting statement")
    severity: str = Field(description="High / Medium / Low")


class ExtractionResult(BaseModel):
    """Complete structured output for one document"""
    doc_id: str
    source_type: SourceType
    functional_requirements: List[FunctionalRequirement] = []
    non_functional_requirements: List[NonFunctionalRequirement] = []
    stakeholders: List[Stakeholder] = []
    decisions: List[Decision] = []
    milestones: List[Milestone] = []
    raw_text_preview: str = Field(description="First 200 chars of input")


class AggregatedReport(BaseModel):
    """Final output across all documents"""
    total_documents: int
    functional_requirements: List[FunctionalRequirement] = []
    non_functional_requirements: List[NonFunctionalRequirement] = []
    stakeholders: List[Stakeholder] = []
    decisions: List[Decision] = []
    milestones: List[Milestone] = []
    conflicts: List[Conflict] = []
    summary: str = Field(description="2-3 sentence executive summary")
