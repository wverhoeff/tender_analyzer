import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Import local backend modules
from agents.supervisor import TenderSupervisor
from workspace.state_machine import TenderState

app = FastAPI(
    title="Civil Engineering Tender Analyzer",
    description="Multi-agent backend using OpenRouter Llama-3 to analyze New Zealand consultancy tenders.",
    version="1.0.0"
)

# Initialize the supervisor
db_dir = "./db/chroma"
ingestion_dir = "./data/ingestion"
os.makedirs(ingestion_dir, exist_ok=True)

supervisor = TenderSupervisor(db_dir=db_dir)

class AnalyzeRequest(BaseModel):
    tender_id: str
    raw_text: str

class ApprovalRequest(BaseModel):
    approved: bool
    feedback: Optional[str] = ""

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Civil Engineering Consultancy Tender Analyzer API",
        "docs": "/docs"
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document (e.g. Markdown tender spec or contract guideline)
    to the ingestion directory and automatically re-index the RAG database.
    """
    if not file.filename.endswith((".md", ".txt")):
        raise HTTPException(status_code=400, detail="Only .md or .txt files are supported for ingestion.")

    file_path = os.path.join(ingestion_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Trigger re-ingestion of Vector Store
        supervisor.vector_store.ingest_markdown()
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": "File uploaded and RAG database successfully updated."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save and ingest file: {str(e)}")

@app.post("/analyze")
def analyze_tender(request: AnalyzeRequest):
    """
    Triggers Phase 1 (Scope Extraction) and Phase 2 (Audit & Estimations) sequentially.
    Puts the workflow into Phase 3 (Approval State) waiting for human sign-off.
    """
    try:
        # Run Phase 1 & Phase 2
        final_state = supervisor.run_complete_workflow(
            tender_id=request.tender_id,
            raw_tender_text=request.raw_text
        )
        return {
            "status": "processing",
            "message": "Phase 1 and Phase 2 completed. Waiting for Phase 3 human approval.",
            "state": final_state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@app.get("/status/{tender_id}")
def get_tender_status(tender_id: str):
    """
    Check the current workflow phase and view intermediate markdown reports.
    """
    state = supervisor.state_machine.load_state(tender_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Tender analysis run '{tender_id}' not found.")
    return state

@app.post("/approve/{tender_id}")
def approve_tender_step(tender_id: str, request: ApprovalRequest):
    """
    Submit approval or trigger a rollback/revision based on manual review.
    """
    try:
        updated_state = supervisor.run_phase_3_approval(
            tender_id=tender_id,
            approved=request.approved,
            feedback=request.feedback
        )
        status_msg = "Tender approved and completed!" if request.approved else "Reverted to Phase 1 for revisions."
        return {
            "status": "success",
            "message": status_msg,
            "state": updated_state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Approval submission failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
