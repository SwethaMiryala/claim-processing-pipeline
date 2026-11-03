from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from graph import graph
from pdf_utils import extract_page_texts

app = FastAPI(title="Claim Processing Pipeline")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/api/process")
async def process_claim(claim_id: str = Form(...), file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")

    pdf_bytes = await file.read()
    try:
        page_texts = extract_page_texts(pdf_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not read PDF: {exc}")
    if not page_texts:
        raise HTTPException(400, "PDF has no pages.")

    try:
        result = await graph.ainvoke({"claim_id": claim_id, "page_texts": page_texts})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Pipeline failed: {exc}")

    return result["final_output"]
