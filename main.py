from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid

from agents.extraction_agent import ExtractionAgent
from agents.conflict_agent import ConflictDetectorAgent
from agents.aggregator_agent import AggregatorAgent
from models.schemas import AggregatedReport, ExtractionResult, SourceType

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Requirements Extraction Agent",
    description="AI agent that extracts structured business requirements from emails, transcripts, and chat logs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Agent initialization (lazy, on first request)
# ─────────────────────────────────────────────
_extractor: Optional[ExtractionAgent] = None
_conflict_detector: Optional[ConflictDetectorAgent] = None
_aggregator: Optional[AggregatorAgent] = None

def get_agents():
    global _extractor, _conflict_detector, _aggregator
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")
    if _extractor is None:
        _extractor = ExtractionAgent(api_key)
        _conflict_detector = ConflictDetectorAgent(api_key)
        _aggregator = AggregatorAgent(api_key)
    return _extractor, _conflict_detector, _aggregator


# ─────────────────────────────────────────────
# Request/Response models
# ─────────────────────────────────────────────
class SingleDocRequest(BaseModel):
    text: str
    doc_id: Optional[str] = None
    source_type: Optional[SourceType] = None


class BatchRequest(BaseModel):
    documents: List[SingleDocRequest]


class PipelineResponse(BaseModel):
    individual_results: List[ExtractionResult]
    aggregated_report: AggregatedReport


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/extract/single", response_model=ExtractionResult)
def extract_single(req: SingleDocRequest):
    """
    Extract structured requirements from a single document.
    Useful for testing and for the frontend's live input panel.
    """
    extractor, _, _ = get_agents()
    doc_id = req.doc_id or f"doc-{uuid.uuid4().hex[:8]}"

    try:
        result = extractor.extract(
            text=req.text,
            doc_id=doc_id,
            source_type=req.source_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/batch", response_model=PipelineResponse)
def extract_batch(req: BatchRequest):
    """
    Full pipeline: extract from multiple documents, detect conflicts, aggregate.
    This is the main hackathon demo endpoint.
    """
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    if len(req.documents) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 documents per batch")

    extractor, conflict_detector, aggregator = get_agents()

    # Step 1: Extract from each document
    results: List[ExtractionResult] = []
    for i, doc in enumerate(req.documents):
        doc_id = doc.doc_id or f"doc-{i+1:03d}"
        try:
            result = extractor.extract(
                text=doc.text,
                doc_id=doc_id,
                source_type=doc.source_type
            )
            results.append(result)
        except Exception as e:
            print(f"[Pipeline] Error extracting {doc_id}: {e}")
            # Continue with other docs rather than failing the whole batch

    if not results:
        raise HTTPException(status_code=500, detail="All document extractions failed")

    # Step 2: Detect conflicts across documents
    conflicts = conflict_detector.detect(results)

    # Step 3: Aggregate into unified report
    report = aggregator.aggregate(results, conflicts)

    return PipelineResponse(
        individual_results=results,
        aggregated_report=report
    )


@app.post("/extract/file", response_model=ExtractionResult)
async def extract_file(file: UploadFile = File(...)):
    """
    Upload a .txt file and extract requirements.
    Accepts plain text files (email exports, transcript dumps).
    """
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only .txt and .md files supported")

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    extractor, _, _ = get_agents()
    doc_id = f"file-{file.filename}"

    try:
        result = extractor.extract(text=text, doc_id=doc_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Run directly for development
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
