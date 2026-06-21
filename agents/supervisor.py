import os
import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import system modules
from workspace.state_machine import TenderStateMachine, TenderState
from agents.auditor import EngineeringAuditor
from agents.estimator import FeeEstimator
from database.vector_store import TenderVectorStore
from database.simple_search import SimpleSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TenderSupervisor:
    """
    Main supervisor agent coordinating the Tender Analysis Workflow:
    - Phase 1: Ingestion & Scope Extraction
    - Phase 2: Regulatory Audit (Auditor) & Fee Estimation (Estimator)
    - Phase 3: Human-in-the-Loop Approval State
    """
    def __init__(self, model_name: str = "meta-llama/llama-3.3-70b-instruct", db_dir: str = "./db/chroma"):
        self.state_machine = TenderStateMachine()
        self.vector_store = TenderVectorStore(persist_directory=db_dir)
        self.search_engine = SimpleSearchEngine()
        
        # Configure OpenRouter LLM connection
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            self.llm = ChatOpenAI(
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                model_name=model_name,
                temperature=0.2
            )
            logger.info("TenderSupervisor successfully initialized with OpenRouter LLM.")
        else:
            self.llm = None
            logger.warning("OPENROUTER_API_KEY environment variable is not set. Running in MOCK mode.")

        # Initialize sub-agents
        self.auditor = EngineeringAuditor(llm=self.llm)
        self.estimator = FeeEstimator(llm=self.llm)

    def _get_database_context(self, scope_text: str) -> str:
        """
        Retrieves context documents from RAG vector store and keyword search fallback.
        """
        context_parts = []
        
        # 1. Similarity Search
        try:
            vector_results = self.vector_store.search_costings(scope_text)
            for i, doc in enumerate(vector_results):
                src = doc.metadata.get("source", "Unknown Source")
                context_parts.append(f"--- RAG Result {i+1} [Source: {src}] ---\n{doc.page_content}")
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")

        # 2. Key Term Fallback Lookups
        key_terms = ["rate", "fee", "cpeng", "acenz", "liability", "retaining", "unitary plan"]
        self.search_engine.load_documents(self._gather_ingested_files_as_docs())
        
        keyword_results = []
        for term in key_terms:
            if term.lower() in scope_text.lower():
                matches = self.search_engine.keyword_search(term)
                for match in matches:
                    text = match.get("content", "")
                    src = match.get("source", "Unknown")
                    if text not in keyword_results:
                        keyword_results.append(f"--- Keyword Search match [{term}] in {src} ---\n{text}")

        context_parts.extend(keyword_results[:3])  # Limit keyword matches to avoid token limits
        
        if not context_parts:
            return "No RAG context or historical project reference data found in local database."
            
        return "\n\n".join(context_parts)

    def _gather_ingested_files_as_docs(self) -> list:
        """Helper to load files from ingestion directory as documents for simple search."""
        docs = []
        ingestion_dir = "./data/ingestion"
        if os.path.exists(ingestion_dir):
            for file_name in os.listdir(ingestion_dir):
                if file_name.endswith(".md") or file_name.endswith(".txt"):
                    filepath = os.path.join(ingestion_dir, file_name)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            docs.append({"source": file_name, "content": f.read()})
                    except Exception as e:
                        logger.error(f"Error loading {file_name}: {str(e)}")
        return docs

    def run_phase_1(self, tender_id: str, raw_tender_text: str) -> TenderState:
        """
        PHASE 1: Ingestion & Scope Extraction
        Parses raw text to identify core deliverables, engineering disciplines required, and timeline.
        """
        logger.info(f"[{tender_id}] Launching Phase 1: Ingestion & Scope Extraction")
        
        # Load or initialize state
        state = self.state_machine.load_state(tender_id)
        if not state:
            state = TenderState(tender_id=tender_id)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert Tender Ingest Agent for a New Zealand Civil Engineering Consultancy.\n"
                "Extract the scope of work from the raw text. Detail:\n"
                "1. Structural elements (e.g. Retaining walls, foundations).\n"
                "2. Civil works (e.g. Stormwater, earthworks, pavements).\n"
                "3. Crucial project parameters (e.g. location, Auckland Council jurisdiction, deadlines).\n"
                "4. Missing information that the consultancy must request.\n\n"
                "Format this beautifully in Markdown."
            )),
            ("user", "Raw Tender Text:\n{raw_text}")
        ])

        if self.llm:
            chain = prompt | self.llm | StrOutputParser()
            try:
                scope_md = chain.invoke({"raw_text": raw_tender_text})
                state.scope_extraction_md = scope_md
            except Exception as e:
                logger.error(f"Phase 1 LLM Execution failed: {str(e)}")
                state.scope_extraction_md = f"### Scope Extraction Error\n{str(e)}"
        else:
            state.scope_extraction_md = f"### Scope Extraction (Mock Mode)\nExtracted basic structural & stormwater layout details for tender {tender_id}."

        state.active_step = "PHASE_2_AUDIT"
        self.state_machine.save_state(state)
        return state

    def run_phase_2(self, tender_id: str) -> TenderState:
        """
        PHASE 2: Regulatory/Risk Audit & Fee Estimation
        Retrieves context and passes task to the Auditor and Estimator agents.
        """
        logger.info(f"[{tender_id}] Launching Phase 2: Risk Audit & Costing Analysis")
        
        state = self.state_machine.load_state(tender_id)
        if not state:
            raise ValueError(f"State not initialized for tender {tender_id}. Run Phase 1 first.")

        # Gather context using RAG
        context_docs = self._get_database_context(state.scope_extraction_md)

        # Trigger Sub-Agents
        logger.info(f"[{tender_id}] Invoking Engineering Auditor sub-agent...")
        state.audit_report_md = self.auditor.audit_risks(
            scope_text=state.scope_extraction_md,
            context_documents=context_docs
        )

        logger.info(f"[{tender_id}] Invoking Fee Estimator sub-agent...")
        state.fee_estimation_md = self.estimator.estimate_fee(
            scope_text=state.scope_extraction_md,
            context_documents=context_docs
        )

        state.active_step = "PHASE_3_APPROVAL"
        self.state_machine.save_state(state)
        return state

    def run_phase_3_approval(self, tender_id: str, approved: bool, feedback: str = "") -> TenderState:
        """
        PHASE 3: Human-in-the-Loop Approval State
        Applies human sign-off feedback to finalize or request revisions.
        """
        logger.info(f"[{tender_id}] Transitioning Phase 3: Applying human approval status")
        state = self.state_machine.load_state(tender_id)
        if not state:
            raise ValueError(f"State not found for tender {tender_id}")

        state.human_approved = approved
        state.revision_feedback = feedback

        if approved:
            state.active_step = "COMPLETED"
            logger.info(f"[{tender_id}] Tender Workflow completed and approved!")
        else:
            # Revert to Phase 1/2 for revisions based on feedback
            state.active_step = "PHASE_1_SCOPE"
            logger.info(f"[{tender_id}] Tender workflow rejected/revision requested. Reverting to PHASE_1_SCOPE.")

        self.state_machine.save_state(state)
        return state

    def run_complete_workflow(self, tender_id: str, raw_tender_text: str) -> TenderState:
        """
        Convenience execution path running Phase 1 and Phase 2 sequentially.
        Workflow automatically halts at Phase 3 waiting for human sign-off.
        """
        self.run_phase_1(tender_id, raw_tender_text)
        return self.run_phase_2(tender_id)
