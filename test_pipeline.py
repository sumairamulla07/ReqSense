"""
Quick test script — run this to verify the pipeline works before the demo.
Usage: GEMINI_API_KEY=your_key python test_pipeline.py
"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(__file__))

from agents.extraction_agent import ExtractionAgent
from agents.conflict_agent import ConflictDetectorAgent
from agents.aggregator_agent import AggregatorAgent

# ── Sample data (realistic Enron-style emails) ──────────────────────────────

EMAIL_1 = """
From: sarah.johnson@enron.com
To: dev-team@enron.com
Subject: Q3 Trading Platform - Requirements Review
Date: Mon, 12 Mar 2001 09:14:22 -0600

Team,

Following yesterday's call with the CTO (Mark Richards), we've finalized the 
core requirements for the Q3 trading platform upgrade.

MUST HAVE by April 15th:
- Real-time price feed integration with latency under 50ms
- User authentication with 2FA support (security audit requires this)
- Audit log for all trades - regulatory compliance mandates this

Should also include:
- Dashboard showing portfolio performance (nice to have for Q3, required Q4)
- CSV export for trade history

The system must handle 10,000 concurrent users without degradation.
Mark has approved the $2M budget. Sarah Chen will lead the frontend team.

Deadline is HARD — April 15 for beta, May 1 for production.

Sarah Johnson
Product Manager
"""

EMAIL_2 = """
From: mike.chen@enron.com
To: sarah.johnson@enron.com, dev-team@enron.com
Subject: RE: Q3 Trading Platform - Requirements Review
Date: Tue, 13 Mar 2001 11:30:00 -0600

Sarah,

Thanks for the summary. A few pushbacks from the engineering side:

1. The 50ms latency requirement for real-time feeds is very aggressive. 
   I'd recommend we target 100ms as the SLA and treat 50ms as a stretch goal.
   
2. Production deadline — our infrastructure team says May 15th is more realistic 
   given the 2FA integration complexity. Can we align with stakeholders on this?

3. We need API rate limiting as a CRITICAL requirement — without it we're 
   exposed to DDoS. This was not in Sarah's list but it's non-negotiable.

4. The concurrent user requirement should be 15,000 not 10,000 — Q2 peak was 
   12,000 and we need headroom.

Mike Chen
Lead Engineer
"""

TRANSCRIPT_1 = """
[Meeting Transcript - Trading Platform Sync - March 14, 2001]

SPEAKER: Mark Richards (CTO)
The April 15 beta deadline is firm. Board presentation is April 20th and we 
need something to show. Non-negotiable.

SPEAKER: Mike Chen (Engineering Lead)  
Understood. We'll make April 15 work for beta but I want May 15 formally 
agreed for production, not May 1. I'm putting this in writing.

SPEAKER: Sarah Johnson (PM)
Agreed on May 15 for production. I'll update the project plan.

SPEAKER: Mark Richards
Also — I want SSO integration added to the requirements. Our enterprise clients 
are asking for it. Make it P2 for now.

SPEAKER: Sarah Johnson
Got it. SSO as P2. I'll add it to the tracker. One more thing — 
the CSV export I mentioned is actually required for our compliance team, 
so it should move to P1.

[End of transcript]
"""


def run_test():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Set GEMINI_API_KEY environment variable first")
        print("   export GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    print("🚀 Initializing agents...")
    extractor = ExtractionAgent(api_key)
    conflict_detector = ConflictDetectorAgent(api_key)
    aggregator = AggregatorAgent(api_key)

    docs = [
        ("email-001", EMAIL_1),
        ("email-002", EMAIL_2),
        ("transcript-001", TRANSCRIPT_1),
    ]

    results = []
    for doc_id, text in docs:
        print(f"\n📄 Extracting: {doc_id}...")
        result = extractor.extract(text=text, doc_id=doc_id)
        results.append(result)
        print(f"   ✅ Found {len(result.functional_requirements)} FRs, "
              f"{len(result.stakeholders)} stakeholders, "
              f"{len(result.milestones)} milestones")

    print("\n🔍 Detecting conflicts...")
    conflicts = conflict_detector.detect(results)
    print(f"   ⚠️  Found {len(conflicts)} conflicts")
    for c in conflicts:
        print(f"   [{c.severity}] {c.type}: {c.description[:80]}...")

    print("\n📊 Aggregating report...")
    report = aggregator.aggregate(results, conflicts)

    print("\n" + "="*60)
    print("FINAL REPORT SUMMARY")
    print("="*60)
    print(f"Documents: {report.total_documents}")
    print(f"Functional Requirements: {len(report.functional_requirements)}")
    print(f"Non-Functional Requirements: {len(report.non_functional_requirements)}")
    print(f"Stakeholders: {len(report.stakeholders)}")
    print(f"Decisions: {len(report.decisions)}")
    print(f"Milestones: {len(report.milestones)}")
    print(f"Conflicts: {len(report.conflicts)}")
    print(f"\nExecutive Summary:\n{report.summary}")

    # Save full output
    with open("test_output.json", "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    print("\n✅ Full output saved to test_output.json")


if __name__ == "__main__":
    run_test()
