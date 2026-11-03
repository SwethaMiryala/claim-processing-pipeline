"""Aggregator: merges every agent's output into the final JSON."""
from state import ClaimState


def aggregator_node(state: ClaimState) -> dict:
    classification = state.get("classification", {})
    final = {
        "claim_id": state["claim_id"],
        "total_pages": len(state.get("page_texts", {})),
        "page_classification": [
            {"page": p, "doc_type": t} for p, t in sorted(classification.items())
        ],
        "identity": state.get("id_result"),
        "discharge_summary": state.get("discharge_result"),
        "itemized_bill": state.get("bill_result"),
    }
    return {"final_output": final}
