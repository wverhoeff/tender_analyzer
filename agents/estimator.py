from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class FeeEstimator:
    """
    Sub-agent focused on professional services fee estimation.
    Estimates:
    - Engineering Design Hours
    - Peer Review / CPEng Sign-off Hours
    - Drafting & Civil 3D Model Creation Hours
    - Matches these roles against consultancy rate matrices to output total costings.
    """
    def __init__(self, llm=None):
        self.llm = llm

    def estimate_fee(self, scope_text: str, context_documents: str) -> str:
        """
        Estimates the professional engineering fees based on project scope and matrix.
        Returns a markdown report showing hourly estimations and calculated totals.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an Expert Estimator for Civil Engineering Consultancy firms.\n"
                "Review the project scope and the context documents (which contain standard rate structures and historical project hours).\n"
                "Provide a detailed professional services fee proposal including:\n"
                "1. Itemized Hour Estimates for roles: Technical Director (CPEng review), Senior CPEng Engineer (design/sign-off), Intermediate Engineer, and Civil 3D Drafter.\n"
                "2. Calculations showing total hours multiplied by the rates found in the context documents.\n"
                "3. Deliverables breakdown (e.g., drawings, reports, consents).\n"
                "4. Contingency fee suggestions based on project complexity.\n\n"
                "Present your findings in a structured Markdown format with tables for hourly calculations."
            )),
            ("user", "Tender Scope:\n{scope_text}\n\nContext Documents (Fee Matrix & History):\n{context_documents}")
        ])

        if self.llm:
            chain = prompt | self.llm | StrOutputParser()
            try:
                report = chain.invoke({
                    "scope_text": scope_text,
                    "context_documents": context_documents
                })
                return report
            except Exception as e:
                return f"### Fee Estimation Error\nAn error occurred during LLM invocation: {str(e)}\n\nFallback: Manual computation required."
        else:
            # Fallback mock report if LLM is not configured / available
            return self._mock_report(scope_text)

    def _mock_report(self, scope_text: str) -> str:
        return f"""# Professional Services Fee Proposal (Mock Fallback)

## Hours & Cost Breakdown
Based on a review of the tender requirements, the following hourly estimates apply:

| Role | Hourly Rate ($) | Estimated Hours | Total Cost ($) |
| :--- | :---: | :---: | :---: |
| Technical Director (CPEng Sign-off) | 300 | 10 | 3,000 |
| Senior Engineer (CPEng Design) | 250 | 40 | 10,000 |
| Intermediate Civil Engineer | 180 | 80 | 14,400 |
| Civil 3D Drafter | 140 | 60 | 8,400 |
| **Total** | | **190** | **$35,800** |

## Deliverables & Tasks
1. **Concept & Detailed Design (60 hours)**: Geometric layout of civil assets.
2. **Drafting / Civil 3D Modeling (60 hours)**: Ground surface modeling and cut/fill cross-sections.
3. **CPEng Peer Review & Producer Statement (PS1) (10 hours)**: Quality review and structural verification.
4. **Project Management & Client Liaison (60 hours)**: Status updates and Auckland Council consenting interface.

## Recommended Contingency
- Suggest **15% contingency ($5,370)** due to potential delays in Auckland Council consent processing.
"""
