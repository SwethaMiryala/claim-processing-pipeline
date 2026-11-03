# Claim Processing Pipeline (FastAPI + LangGraph)

Upload a claim PDF → an AI **Segregator** classifies each page into 9 document types →
only the relevant pages are sent to 3 extraction agents → an **Aggregator** returns one JSON.

```
START → Segregator (LLM) ─┬→ ID Agent ────────────┐
                          ├→ Discharge Agent ─────┼→ Aggregator → END
                          └→ Itemized Bill Agent ─┘
```

## How it works
- **Segregator** (`app/nodes/segregator.py`): extracts text per page, sends short previews to Gemini with a
  structured-output schema, gets `page → doc_type` for the 9 types, then builds `agent_pages`
  (`identity_document` + `claim_forms` → ID agent, `discharge_summary` → Discharge agent,
  `itemized_bill` → Bill agent). Other types are classified but not extracted.
- **Routing** (`app/graph.py`): a conditional edge runs only the agents that received pages (in parallel).
- **Agents**: each receives only the text of its own pages (never the whole PDF) and returns structured JSON.
  The bill agent sums line items **in Python** to get `total_amount`.
- **Aggregator**: merges everything into the response.

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # Mac/Linux: cp .env.example .env  → then add your GOOGLE_API_KEY
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs and try `POST /api/process`, or:
```bash
curl -X POST http://127.0.0.1:8000/api/process -F "claim_id=CLM-001" -F "file=@final.pdf"
```

## Limitations
Text-based PDFs only (scanned pages would need OCR / a vision model).
