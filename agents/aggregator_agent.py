import google.genai as genai
from typing import List
from models.schemas import ExtractionResult, AggregatedReport, Stakeholder, Conflict


SUMMARY_PROMPT = """You are a project manager. Based on the following extracted project data, write a concise executive summary in exactly 2-3 sentences. Focus on: what the project is about, key deadline or milestone, and most critical requirement.

DATA:
{data}

Return only the summary text, no formatting.
"""


class AggregatorAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def _merge_stakeholders(self, all_results: List[ExtractionResult]) -> List[Stakeholder]:
        """Merge and deduplicate stakeholders across documents, summing mention counts"""
        merged = {}
        for result in all_results:
            for s in result.stakeholders:
                key = s.name.lower().strip()
                if key in merged:
                    merged[key].mentions += s.mentions
                    # Upgrade authority if we find a higher one
                    authority_rank = {"Approver": 3, "Influencer": 2, "Informed": 1}
                    if authority_rank.get(s.decision_authority, 0) > authority_rank.get(merged[key].decision_authority, 0):
                        merged[key].decision_authority = s.decision_authority
                    if s.email and not merged[key].email:
                        merged[key].email = s.email
                else:
                    merged[key] = s.model_copy()
        return sorted(merged.values(), key=lambda x: -x.mentions)

    def _generate_summary(self, report: AggregatedReport) -> str:
        """Use Gemini to generate a brief executive summary"""
        data_lines = [
            f"Documents analyzed: {report.total_documents}",
            f"Functional requirements: {len(report.functional_requirements)}",
            f"NFRs: {len(report.non_functional_requirements)}",
            f"Stakeholders: {', '.join(s.name for s in report.stakeholders[:5])}",
            f"Key milestones: {', '.join(f'{m.name} ({m.deadline})' for m in report.milestones[:3])}",
            f"Conflicts detected: {len(report.conflicts)}",
            f"P1 requirements: {', '.join(fr.description[:60] for fr in report.functional_requirements if fr.priority == 'P1')[:300]}",
        ]
        prompt = SUMMARY_PROMPT.format(data="\n".join(data_lines))
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"temperature": 0.3}
            )
            return response.text.strip()
        except Exception:
            return f"Analysis of {report.total_documents} document(s) identified {len(report.functional_requirements)} functional requirements and {len(report.conflicts)} conflicts."

    def aggregate(self, results: List[ExtractionResult], conflicts: List[Conflict]) -> AggregatedReport:
        """Merge all extraction results into a single unified report"""

        # Counter for re-numbering IDs globally
        fr_list, nfr_list, dec_list, ms_list = [], [], [], []
        fr_count = nfr_count = dec_count = ms_count = 1

        for result in results:
            for fr in result.functional_requirements:
                fr.id = f"FR-{fr_count:03d}"
                fr_count += 1
                fr_list.append(fr)

            for nfr in result.non_functional_requirements:
                nfr.id = f"NFR-{nfr_count:03d}"
                nfr_count += 1
                nfr_list.append(nfr)

            for dec in result.decisions:
                dec.id = f"DEC-{dec_count:03d}"
                dec_count += 1
                dec_list.append(dec)

            for ms in result.milestones:
                ms.id = f"MS-{ms_count:03d}"
                ms_count += 1
                ms_list.append(ms)

        # Sort by priority
        priority_order = {"P1": 0, "P2": 1, "P3": 2}
        fr_list.sort(key=lambda x: priority_order.get(x.priority, 3))
        nfr_list.sort(key=lambda x: priority_order.get(x.priority, 3))

        merged_stakeholders = self._merge_stakeholders(results)

        report = AggregatedReport(
            total_documents=len(results),
            functional_requirements=fr_list,
            non_functional_requirements=nfr_list,
            stakeholders=merged_stakeholders,
            decisions=dec_list,
            milestones=ms_list,
            conflicts=conflicts,
            summary=""
        )

        report.summary = self._generate_summary(report)
        return report
