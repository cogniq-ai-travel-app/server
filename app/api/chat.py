from fastapi import APIRouter, HTTPException, Request, UploadFile
import json
import base64
from app.models.schemas import ChatRequest, PicoResponseSchema
from app.agent.graph import pico_agent_executor
import copy

router = APIRouter()

@router.post("/api/chat", response_model=PicoResponseSchema)
async def handle_pico_chat_endpoint(request: Request):
    """
    Main communication interface endpoint for the PackPals mobile client.
    Now supports BOTH application/json (text/images) and multipart/form-data (PDFs/Files).
    """
    try:
        content_type = request.headers.get("content-type", "")

        if "multipart/form-data" in content_type:
            form = await request.form()
            payload_str = form.get("payload_data")
            file: UploadFile = form.get("file")
            
            if not payload_str:
                raise HTTPException(status_code=400, detail="Missing payload_data in form")
                
            request_dict = json.loads(payload_str)
            
            if file:
                file_bytes = await file.read()
                request_dict["attachment"] = {
                    "base64_data": base64.b64encode(file_bytes).decode("utf-8"),
                    "mime_type": file.content_type or "application/octet-stream"
                }
            else:
                request_dict["attachment"] = None
                
        elif "application/json" in content_type:
            request_dict = await request.json()
        else:
            raise HTTPException(status_code=415, detail="Unsupported Media Type")


        validated_request = ChatRequest(**request_dict)
        final_request_dict = validated_request.model_dump()

        print("\n" + "="*60)
        print("🚨 [RAW FRONTEND PAYLOAD RECEIVED OVER THE WIRE] 🚨")
        print(f"• Content Type: {content_type}")

        log_dict = copy.deepcopy(final_request_dict)
        if log_dict.get("attachment") and log_dict["attachment"].get("base64_data"):
            log_dict["attachment"]["base64_data"] = "[BASE64_DATA_TRUNCATED_FOR_LOGS]"

        if log_dict.get("current_list"):
            log_dict["current_list"] = f"[TRUNCATED: {len(log_dict['current_list'])} items]"
            
        if log_dict.get("trip_context") and log_dict["trip_context"].get("categories"):
            log_dict["trip_context"]["categories"] = "[TRUNCATED CATEGORIES]"

        print(f"• Raw Body Data:\n{json.dumps(log_dict, indent=4)}")


        initial_state = {
            "request_payload": final_request_dict, 
            "internal_logs": [],
            "final_response": {}
        }
        
        config = {"configurable": {"thread_id": validated_request.thread_id}}
        
        output_state = pico_agent_executor.invoke(initial_state, config=config)
        
        final_json_payload = output_state.get("final_response")
        
        if not final_json_payload:
            raise HTTPException(
                status_code=500, 
                detail="The internal LangGraph orchestration pipeline failed to compile a valid final response state."
            )
            
        return final_json_payload

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(f"\n[CRITICAL ROUTE RUNTIME EXCEPTION]: {str(e)}\n")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Agentic Server Exception: {str(e)}"
        )