from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # 🌟 NEW: Import the checkpointer memory module
from app.agent.state import AgentState
from app.agent.nodes import (
    handle_active_trip_node,
    handle_new_trip_wizard_node,
    handle_general_travel_node
)

def route_incoming_request(state: AgentState) -> str:
    """
    Evaluates the properties of the incoming validation state request payload 
    to selectively route traffic to the specialized processing nodes based on structural intent.
    """
    payload = state.get("request_payload")
    
    if not payload:
        return "general_node"
        
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()

    ctx = payload.get("trip_context") or {}

    mode = payload.get("mode", "general")

    destination = ctx.get("destination") if isinstance(ctx, dict) else getattr(ctx, "destination", None)

    if mode == "newTrip":
        return "new_trip_node"
        
    if destination or mode == "activeTrips":
        return "active_trip_node"
        
    return "general_node"



workflow = StateGraph(AgentState)

workflow.add_node("active_trip_node", handle_active_trip_node)
workflow.add_node("new_trip_node", handle_new_trip_wizard_node)
workflow.add_node("general_node", handle_general_travel_node)

workflow.set_conditional_entry_point(
    route_incoming_request,
    {
        "active_trip_node": "active_trip_node",
        "new_trip_node": "new_trip_node",
        "general_node": "general_node"
    }
)

workflow.add_edge("active_trip_node", END)
workflow.add_edge("new_trip_node", END)
workflow.add_edge("general_node", END)

memory = MemorySaver()

pico_agent_executor = workflow.compile(checkpointer=memory)