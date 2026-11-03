from typing import Dict, List, TypedDict


class ClaimState(TypedDict, total=False):
    claim_id: str
    page_texts: Dict[int, str]          # page -> text
    classification: Dict[int, str]      # page -> one of the 9 doc types
    agent_pages: Dict[str, List[int]]   # agent node name -> pages routed to it
    id_result: dict
    discharge_result: dict
    bill_result: dict
    final_output: dict
