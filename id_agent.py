"""ID Agent: identity info + policy details. Sees ONLY its assigned pages."""
from typing import List, Optional

from pydantic import BaseModel

from llm import extract
from state import ClaimState


class IdNumber(BaseModel):
    type: str
    value: str


class IdentityInfo(BaseModel):
    patient_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    contact_number: Optional[str] = None
    email: Optional[str] = None
    id_numbers: List[IdNumber] = []
    policy_number: Optional[str] = None
    group_number: Optional[str] = None
    insurer: Optional[str] = None
    claim_reference: Optional[str] = None


SYSTEM = """Extract identity information from the pages given.
Include patient name, date of birth, gender, address, contact details, every ID number
(with its type, e.g. 'government ID', 'policy number', 'patient ID') and policy details.
Use null for anything not present. Never invent values."""


def id_agent_node(state: ClaimState) -> dict:
    pages = state["agent_pages"]["id_agent"]
    text = "\n\n".join(f"--- PAGE {p} ---\n{state['page_texts'][p]}" for p in pages)
    result = extract(IdentityInfo, SYSTEM, text)
    return {"id_result": {"pages_processed": pages, **result.model_dump()}}
