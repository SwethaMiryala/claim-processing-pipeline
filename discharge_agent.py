"""Discharge Summary Agent. Sees ONLY its assigned pages."""
from typing import List, Optional

from pydantic import BaseModel

from llm import extract
from state import ClaimState


class DischargeInfo(BaseModel):
    patient_name: Optional[str] = None
    hospital: Optional[str] = None
    diagnosis: Optional[str] = None
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    attending_physician: Optional[str] = None
    discharge_medications: List[str] = []
    follow_up: Optional[str] = None


SYSTEM = """Extract data from a hospital discharge summary: diagnosis, admission date,
discharge date, physician details, hospital, discharge medications and follow-up.
Use null when missing. Never invent values."""


def discharge_agent_node(state: ClaimState) -> dict:
    pages = state["agent_pages"]["discharge_agent"]
    text = "\n\n".join(f"--- PAGE {p} ---\n{state['page_texts'][p]}" for p in pages)
    result = extract(DischargeInfo, SYSTEM, text)
    return {"discharge_result": {"pages_processed": pages, **result.model_dump()}}
