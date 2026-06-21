import os
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class TenderState(BaseModel):
    tender_id: str
    active_step: str = "PHASE_1_SCOPE"  # PHASE_1_SCOPE, PHASE_2_AUDIT, PHASE_3_APPROVAL, COMPLETED
    scope_extraction_md: str = ""
    audit_report_md: str = ""
    fee_estimation_md: str = ""
    human_approved: Optional[bool] = None  # None = Pending, True = Approved, False = Needs Revision
    revision_feedback: str = ""

class TenderStateMachine:
    """
    Manages the active state machine state for a tender analysis run.
    Persists data as JSON and writes intermediary markdown reports to the workspace folder.
    """
    def __init__(self, workspace_dir: str = "./workspace"):
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)

    def _get_state_filepath(self, tender_id: str) -> str:
        return os.path.join(self.workspace_dir, f"{tender_id}_state.json")

    def _get_markdown_filepath(self, tender_id: str, phase: str) -> str:
        return os.path.join(self.workspace_dir, f"{tender_id}_{phase}.md")

    def load_state(self, tender_id: str) -> Optional[TenderState]:
        """
        Loads the saved state for a tender_id if it exists.
        """
        filepath = self._get_state_filepath(tender_id)
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return TenderState(**data)

    def save_state(self, state: TenderState):
        """
        Saves the structured state as JSON and updates intermediary markdown files.
        """
        filepath = self._get_state_filepath(state.tender_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state.model_dump(), f, indent=4)

        # Write out separate intermediary markdown documents for easy review
        if state.scope_extraction_md:
            self.write_markdown(state.tender_id, "phase1_scope", state.scope_extraction_md)
        if state.audit_report_md:
            self.write_markdown(state.tender_id, "phase2_audit", state.audit_report_md)
        if state.fee_estimation_md:
            self.write_markdown(state.tender_id, "phase2_fee_estimation", state.fee_estimation_md)

    def write_markdown(self, tender_id: str, phase: str, content: str):
        """
        Helper to write standard Markdown documents for a specific phase.
        """
        filepath = self._get_markdown_filepath(tender_id, phase)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def delete_state(self, tender_id: str):
        """
        Cleans up the saved state files for a tender_id.
        """
        filepath = self._get_state_filepath(tender_id)
        if os.path.exists(filepath):
            os.remove(filepath)
        for phase in ["phase1_scope", "phase2_audit", "phase2_fee_estimation"]:
            md_path = self._get_markdown_filepath(tender_id, phase)
            if os.path.exists(md_path):
                os.remove(md_path)
