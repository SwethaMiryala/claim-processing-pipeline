"""Segregator Agent (AI): classifies every page into one of 9 document types
and decides which pages go to which extraction agent."""
from typing import Dict, List, Literal

from pydantic import BaseModel

from llm import extract
from state import ClaimState

DocType = Literal[
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]
DOC_TYPES = list(DocType.__args__)

# doc type -> extraction agent node (only 3 agents exist)
ROUTING = {
    "identity_document": "id_agent",
    "claim_forms": "id_agent",          # policy details live on the claim form
    "discharge_summary": "discharge_agent",
    "itemized_bill": "itemized_bill_agent",
}
AGENT_NODES = ["id_agent", "discharge_agent", "itemized_bill_agent"]


class PageLabel(BaseModel):
    page: int
    doc_type: DocType


class Segregation(BaseModel):
    pages: List[PageLabel]


SYSTEM = f"""You classify pages of a medical insurance claim PDF.
Assign EXACTLY ONE document type to every page. Allowed types: {DOC_TYPES}.

Guidance:
- claim_forms: insurance claim form (claim reference, amount claimed, declaration)
- cheque_or_bank_details: cheque / bank account details
- identity_document: government ID card, driving licence, passport
- itemized_bill: hospital or pharmacy bill listing items with qty/rate/amount
- discharge_summary: hospital discharge summary (admission/discharge dates, diagnosis)
- prescription: doctor's prescription (Rx)
- investigation_report: lab / pathology / radiology reports
- cash_receipt: payment receipt
- other: anything else (consent forms, letters, questionnaires, referral letters, registration forms)
Return one entry per page, using the page numbers given."""


def _keyword_fallback(text: str) -> str:
    """Cheap backup classifier used only if the LLM call fails."""
    t = text.lower()
    rules = [
        ("discharge_summary", ["discharge summary"]),
        ("itemized_bill", ["itemized", "invoice number", "bill number"]),
        ("cash_receipt", ["cash receipt"]),
        ("identity_document", ["government id", "id number", "driving licence", "passport"]),
        ("cheque_or_bank_details", ["cheque", "bank account"]),
        ("claim_forms", ["medical claim form", "claim reference"]),
        ("prescription", ["prescription", "rx "]),
        ("investigation_report", ["laboratory report", "lab report", "reference range"]),
    ]
    for label, keys in rules:
        if any(k in t for k in keys):
            return label
    return "other"


def segregator_node(state: ClaimState) -> dict:
    page_texts: Dict[int, str] = state["page_texts"]

    # Only a short preview of each page is needed to classify it.
    listing = "\n\n".join(
        f"=== PAGE {n} ===\n{text[:1200] or '[no extractable text]'}"
        for n, text in page_texts.items()
    )

    labels: Dict[int, str] = {}
    try:
        result = extract(Segregation, SYSTEM, listing)
        labels = {p.page: p.doc_type for p in result.pages}
    except Exception as exc:  # noqa: BLE001
        print(f"[segregator] LLM failed ({exc}); using keyword fallback")

    classification = {
        n: labels.get(n) or _keyword_fallback(text) for n, text in page_texts.items()
    }

    agent_pages: Dict[str, List[int]] = {name: [] for name in AGENT_NODES}
    for page, doc_type in classification.items():
        agent = ROUTING.get(doc_type)
        if agent:
            agent_pages[agent].append(page)

    return {"classification": classification, "agent_pages": agent_pages}
