from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class EngineeringAuditor:
    """
    Sub-agent focused on auditing tender text for engineering/consultancy risks.
    Analyzes:
    - Auckland Council / Unitary Plan zoning and consent constraints.
    - Required CPEng (Chartered Professional Engineer) sign-offs.
    - Producer Statement commitments (PS1 Design, PS2 Design Review, PS4 Construction Review).
    - ACENZ (Association of Consulting Engineers NZ) liability limitation clauses.
    """
    def __init__(self, llm=None):
        self.llm = llm

    def audit_risks(self, scope_text: str, context_documents: str) -> str:
        """
        Audits the scope text against engineering risks and standard guidelines.
        Returns a markdown report summarizing risks and recommendations.
        """
        # Audit Prompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a Senior Engineering Auditor specialized in New Zealand Civil Engineering Consultancy contracts.\n"
                "Analyze the provided tender scope and context documents. Identify potential risks relating to:\n"
                "1. Auckland Council / Auckland Unitary Plan consent constraints.\n"
                "2. CPEng sign-off requirements.\n"
                "3. Producer Statement (PS1/PS2/PS4) commitments.\n"
                "4. ACENZ liability limitation clauses (e.g. limit of liability matching standard terms, capping total liability).\n\n"
                "Present your findings in a structured Markdown format with:\n"
                "- Executive Risk Summary\n"
                "- Specific Risks Identified (categorized with risk levels: High, Medium, Low)\n"
                "- Mitigation Recommendations\n"
                "- Mandatory CPEng/PS1/PS4 Sign-offs"
            )),
            ("user", "Tender Scope:\n{scope_text}\n\nContext Documents (ACENZ rules & past project data):\n{context_documents}")
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
                return f"### Risk Audit Error\nAn error occurred during LLM invocation: {str(e)}\n\nFallback: Manual review required."
        else:
            # Fallback mock report if LLM is not configured / available
            return self._mock_report(scope_text)

    def _mock_report(self, scope_text: str) -> str:
        return f"""# Engineering Risk Audit Report (Mock Fallback)

## Executive Risk Summary
A review of the tender scope shows potential compliance and liability exposures regarding structural and civil engineering sign-offs in Auckland.

## Specific Risks Identified
1. **Auckland Unitary Plan Consent Constraints [MEDIUM RISK]**
   - *Detail*: Earthworks and retaining wall height constraints might trigger resource consent triggers under the Auckland Unitary Plan.
2. **CPEng Sign-off Requirements [HIGH RISK]**
   - *Detail*: Auckland Council requires specific Chartered Professional Engineer (CPEng) sign-offs for structural designs and retaining wall stability.
3. **Producer Statement Commitments [HIGH RISK]**
   - *Detail*: Scope requires PS1 (Design) and PS4 (Construction Review). This binds the consultancy to construction observation schedules.
4. **ACENZ Liability Caps [MEDIUM RISK]**
   - *Detail*: The tender document does not explicitly state a cap on liability. Standard ACENZ caps (e.g., 5x fee or $200k, whichever is greater) must be negotiated.

## Mitigation Recommendations
- Request modification of standard terms to insert an ACENZ liability cap.
- Ensure that the budget accounts for construction observation hours needed for the PS4 sign-off.
"""
