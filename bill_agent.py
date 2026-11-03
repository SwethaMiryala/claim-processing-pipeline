"""Itemized Bill Agent: extracts every line item and CALCULATES the total in code
(we don't trust the LLM to do arithmetic). Sees ONLY its assigned pages."""
from typing import List, Optional

from pydantic import BaseModel

from llm import extract
from state import ClaimState


class BillItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    rate: Optional[float] = None
    amount: float


class Bill(BaseModel):
    bill_number: Optional[str] = None
    bill_date: Optional[str] = None
    items: List[BillItem] = []
    stated_total: Optional[float] = None  # total printed on the document (tax etc. included)


class BillExtraction(BaseModel):
    bills: List[Bill] = []


SYSTEM = """Extract ALL line items from the itemized bill(s) given: description, quantity,
rate and line amount (numbers only, no currency symbols). Do NOT include subtotal, tax,
discount, or total rows as items. Put the grand total printed on each bill in stated_total.
If the pages contain several bills, return each one separately."""


def bill_agent_node(state: ClaimState) -> dict:
    pages = state["agent_pages"]["itemized_bill_agent"]
    text = "\n\n".join(f"--- PAGE {p} ---\n{state['page_texts'][p]}" for p in pages)
    result = extract(BillExtraction, SYSTEM, text)

    bills = []
    grand_total = 0.0
    for b in result.bills:
        items_total = round(sum(i.amount for i in b.items), 2)
        grand_total += items_total
        bills.append({**b.model_dump(), "calculated_items_total": items_total})

    return {
        "bill_result": {
            "pages_processed": pages,
            "bills": bills,
            "total_amount": round(grand_total, 2),  # sum of all line items
        }
    }
