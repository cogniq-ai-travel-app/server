from typing import TypedDict, Any

class AgentState(TypedDict):

    request_payload: dict[str, Any] 
    internal_logs: list[Any]  
    current_draft: dict[str, Any]
    final_response: dict[str, Any]