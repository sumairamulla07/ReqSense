from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import os, uuid, base64, json

from agents.extraction_agent import ExtractionAgent
from agents.conflict_agent import ConflictDetectorAgent
from agents.aggregator_agent import AggregatorAgent
from models.schemas import AggregatedReport, ExtractionResult, SourceType

app = FastAPI(title="ReqSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Agent init ────────────────────────────────────────────
_extractor = _conflict_detector = _aggregator = None

def get_agents():
    global _extractor, _conflict_detector, _aggregator
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    if _extractor is None:
        _extractor = ExtractionAgent(api_key)
        _conflict_detector = ConflictDetectorAgent(api_key)
        _aggregator = AggregatorAgent(api_key)
    return _extractor, _conflict_detector, _aggregator

# ── Gmail OAuth state (in-memory for hackathon) ───────────
gmail_state = {}   # holds flow + credentials between requests

# ── Request models ────────────────────────────────────────
class SingleDocRequest(BaseModel):
    text: str
    doc_id: Optional[str] = None
    source_type: Optional[SourceType] = None

class BatchRequest(BaseModel):
    documents: List[SingleDocRequest]

class PipelineResponse(BaseModel):
    individual_results: List[ExtractionResult]
    aggregated_report: AggregatedReport

# ── Core routes ───────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/extract/single", response_model=ExtractionResult)
def extract_single(req: SingleDocRequest):
    extractor, _, _ = get_agents()
    doc_id = req.doc_id or f"doc-{uuid.uuid4().hex[:8]}"
    try:
        return extractor.extract(text=req.text, doc_id=doc_id, source_type=req.source_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract/batch", response_model=PipelineResponse)
def extract_batch(req: BatchRequest):
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    extractor, conflict_detector, aggregator = get_agents()
    results = []
    for i, doc in enumerate(req.documents):
        doc_id = doc.doc_id or f"doc-{i+1:03d}"
        try:
            results.append(extractor.extract(text=doc.text, doc_id=doc_id, source_type=doc.source_type))
        except Exception as e:
            print(f"[Pipeline] Error on {doc_id}: {e}")
    if not results:
        raise HTTPException(status_code=500, detail="All extractions failed")
    conflicts = conflict_detector.detect(results)
    report = aggregator.aggregate(results, conflicts)
    return PipelineResponse(individual_results=results, aggregated_report=report)

@app.post("/extract/file", response_model=ExtractionResult)
async def extract_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    extractor, _, _ = get_agents()
    try:
        return extractor.extract(text=text, doc_id=f"file-{file.filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Gmail OAuth routes ────────────────────────────────────
@app.get("/auth/gmail/start")
def gmail_start():
    try:
        from google_auth_oauthlib.flow import Flow
        import os as _os
        _os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        creds_file = 'gmail_credentials.json'
        if not os.path.exists(creds_file):
            raise HTTPException(status_code=500, detail="gmail_credentials.json not found")

        with open(creds_file) as f:
            client_config = json.load(f)

        # Build auth URL manually without PKCE so state alone is enough
        client_data = client_config.get('web') or client_config.get('installed')
        client_id = client_data['client_id']
        client_secret = client_data['client_secret']

        import secrets
        state = secrets.token_urlsafe(16)

        auth_url = (
            "https://accounts.google.com/o/oauth2/v2/auth"
            f"?client_id={client_id}"
            "&redirect_uri=http%3A%2F%2Flocalhost%3A8000%2Fauth%2Fgmail%2Fcallback"
            "&response_type=code"
            "&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fgmail.readonly"
            f"&state={state}"
            "&access_type=offline"
            "&prompt=select_account"
        )

        # Save what we need for the callback
        gmail_state['state'] = state
        gmail_state['client_id'] = client_id
        gmail_state['client_secret'] = client_secret
        gmail_state['token_uri'] = client_data.get('token_uri', 'https://oauth2.googleapis.com/token')

        return {"auth_url": auth_url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/gmail/callback")
def gmail_callback(code: str = None, state: str = None, error: str = None):
    if error:
        return HTMLResponse(f"""<html><body style="background:#0a0a14;color:#f87171;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
        <div style="text-align:center"><h2>Access Denied</h2><p>{error}</p>
        <p style="color:#6b6b8a;font-size:13px">Add your email as a Test User in Google Cloud Console → OAuth consent screen</p>
        <button onclick="window.close()" style="margin-top:16px;padding:8px 20px;background:#7c6dfa;color:white;border:none;border-radius:8px;cursor:pointer;font-size:14px">Close</button>
        </div></body></html>""")
    try:
        import requests as req_lib
        import os as _os
        _os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

        # Exchange authorization code for tokens manually
        token_response = req_lib.post(
            gmail_state.get('token_uri', 'https://oauth2.googleapis.com/token'),
            data={
                'code': code,
                'client_id': gmail_state['client_id'],
                'client_secret': gmail_state['client_secret'],
                'redirect_uri': 'http://localhost:8000/auth/gmail/callback',
                'grant_type': 'authorization_code',
            }
        )
        token_data = token_response.json()

        if 'error' in token_data:
            raise Exception(token_data.get('error_description', token_data['error']))

        # Store as plain dict — no pickle needed
        gmail_state['creds'] = {
            'token': token_data.get('access_token'),
            'refresh_token': token_data.get('refresh_token'),
            'token_uri': gmail_state.get('token_uri', 'https://oauth2.googleapis.com/token'),
            'client_id': gmail_state['client_id'],
            'client_secret': gmail_state['client_secret'],
            'scopes': ['https://www.googleapis.com/auth/gmail.readonly'],
        }

        return HTMLResponse("""<html><head><title>Gmail Connected</title></head>
        <body style="background:#0a0a14;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
        <div style="text-align:center">
          <div style="font-size:56px;margin-bottom:12px">✓</div>
          <h2 style="margin:0 0 8px;color:#e2e2f0">Gmail Connected!</h2>
          <p style="color:#6b6b8a;margin:0 0 20px;font-size:13px">You can close this tab and return to ReqSense.</p>
          <script>
            if(window.opener){window.opener.postMessage('gmail_auth_success','*');setTimeout(()=>window.close(),800)}
          </script>
        </div></body></html>""")
    except Exception as e:
        return HTMLResponse(f"""<html><body style="background:#0a0a14;color:#f87171;font-family:sans-serif;padding:40px">
        <h2>Error</h2><pre style="color:#fca5a5">{e}</pre>
        <button onclick="window.close()" style="padding:8px 16px;background:#7c6dfa;color:white;border:none;border-radius:8px;cursor:pointer;margin-top:16px">Close</button>
        </body></html>""")

@app.get("/gmail/fetch")
def gmail_fetch(max_results: int = 10, query: str = ""):
    creds_data = gmail_state.get('creds')
    if not creds_data:
        raise HTTPException(status_code=401, detail="Not authenticated with Gmail")
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        creds = Credentials(
            token=creds_data['token'],
            refresh_token=creds_data['refresh_token'],
            token_uri=creds_data['token_uri'],
            client_id=creds_data['client_id'],
            client_secret=creds_data['client_secret'],
            scopes=creds_data['scopes'],
        )
        service = build('gmail', 'v1', credentials=creds)
        params = {'userId': 'me', 'maxResults': max_results}
        if query:
            params['q'] = query
        results = service.users().messages().list(**params).execute()
        messages = results.get('messages', [])
        emails = []
        for msg in messages:
            try:
                full = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
                headers = {h['name'].lower(): h['value'] for h in full['payload'].get('headers', [])}
                body = _extract_body(full['payload'])
                formatted = f"""From: {headers.get('from', 'Unknown')}
To: {headers.get('to', 'Unknown')}
Date: {headers.get('date', 'Unknown')}
Subject: {headers.get('subject', '(no subject)')}

{body[:4000]}"""
                emails.append({
                    "id": msg['id'],
                    "subject": headers.get('subject', '(no subject)'),
                    "from": headers.get('from', ''),
                    "date": headers.get('date', '')[:25] if headers.get('date') else '',
                    "formatted_text": formatted
                })
            except Exception as e:
                print(f"Skipping {msg['id']}: {e}")
        return {"emails": emails, "count": len(emails)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _extract_body(payload) -> str:
    """Recursively extract plain text from Gmail payload."""
    body = ""
    if payload.get('mimeType') == 'text/plain':
        data = payload.get('body', {}).get('data', '')
        if data:
            body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    elif 'parts' in payload:
        for part in payload['parts']:
            body = _extract_body(part)
            if body:
                break
    # Strip excessive whitespace
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    return '\n'.join(lines)

@app.get("/gmail/status")
def gmail_status():
    return {"connected": 'creds' in gmail_state}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

