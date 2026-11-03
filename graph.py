"""LangGraph workflow:

START -> segregator -> (id_agent | discharge_agent | itemized_bill_agent) -> aggregator -> END
Only the agents that actually received pages are run.
"""
from langgraph.graph import END, START, StateGraph

from aggregator import aggregator_node
from bill_agent import bill_agent_node
from discharge_agent import discharge_agent_node
from id_agent import id_agent_node
from segregator import AGENT_NODES, segregator_node
from state import ClaimState


def route_after_segregator(state: ClaimState):
    targets = [name for name in AGENT_NODES if state["agent_pages"].get(name)]
    return targets or ["aggregator"]  # nothing relevant -> skip straight to aggregator


def build_graph():
    g = StateGraph(ClaimState)
    g.add_node("segregator", segregator_node)
    g.add_node("id_agent", id_agent_node)
    g.add_node("discharge_agent", discharge_agent_node)
    g.add_node("itemized_bill_agent", bill_agent_node)
    g.add_node("aggregator", aggregator_node)

    g.add_edge(START, "segregator")
    g.add_conditional_edges(
        "segregator", route_after_segregator, AGENT_NODES + ["aggregator"]
    )
    for name in AGENT_NODES:
        g.add_edge(name, "aggregator")
    g.add_edge("aggregator", END)
    return g.compile()


graph = build_graph()
