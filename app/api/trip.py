import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from google import genai
from google.genai import types

from app.core.config import settings
from app.models.schemas import PicoResponseSchema
from app.agent.nodes import generate_json_with_fallback

router = APIRouter()
client = genai.Client(api_key=settings.GOOGLE_API_KEY)


class GenerateTripRequest(BaseModel):
    destination: str
    tripVibe: str
    packingStyle: str
    startDate: str
    endDate: str
    durationDays: int
    fromLocation: str
    activities: List[str] = []
    customActivities: List[str] = []
    baseline_categories: List[Dict[str, Any]] = [] 

@router.post("/api/trip/generate")
async def generate_trip_endpoint(request: GenerateTripRequest):
    try:
        print(f"🚀 [TRIP EDITOR] Reviewing and enhancing list for {request.destination}", flush=True)
        
        all_activities = request.activities + request.customActivities

        prompt_text = f"""
        You are Pico, an expert travel packing assistant. 
        My internal rules engine has generated a "Baseline Packing List" for the user. 
        Your job is to act as the final Editor. You must read the Baseline List, remove unnecessary items, and add highly specific, localized items based on the trip details.

        --- TRIP DETAILS ---
        Destination: {request.destination} (Coming from {request.fromLocation})
        Duration: {request.durationDays} days ({request.startDate} to {request.endDate})
        Vibe: {request.tripVibe}
        Style: {request.packingStyle} 
        Activities: {', '.join(all_activities) if all_activities else 'None specified'}

        --- BASELINE PACKING LIST (From Rules Engine) ---
        {json.dumps(request.baseline_categories, indent=2)}

        --- YOUR MISSION ---
        1. KEEP standard items from the Baseline List that make sense. Adjust their 'quantity' to perfectly match the {request.durationDays} day duration.
        2. REMOVE items from the Baseline List that do not fit the specific destination, weather, or 'packingStyle' (e.g., if Style is 'light', aggressively remove 'optional' items).
        3. ADD new, highly specific items that the Rules Engine missed (e.g., if going to a rainy destination, add an umbrella; if going to a specific country, add the correct power adapter type).
        4. Organize the final items into logical categories.

        CRITICAL OUTPUT RULES:
        - Return ONLY a valid JSON object matching the 'updated_draft' section of the PicoResponseSchema.
        - You must populate the 'categories' array inside 'updated_draft' with the final, edited list.
        - Set 'suggestionAction' to type 'none'.
        - FORMATTING STRICT RULE: When generating the 'name' field for an item, provide ONLY the raw item name. Do NOT prefix the string with "name: " or any other label (e.g., output "Deodorant", NOT "name: Deodorant").
        """


        final_dict = generate_json_with_fallback(
            contents=prompt_text,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=PicoResponseSchema,
            ),
        )
        
        if not final_dict or not final_dict.get("updated_draft"):
             raise ValueError("AI failed to generate a valid edited list.")
             
        merged_draft = request.model_dump()
        merged_draft["categories"] = final_dict["updated_draft"].get("categories", [])

        return {"updated_draft": merged_draft}

    except Exception as e:
        print(f"\n[TRIP GENERATION ERROR]: {str(e)}\n", flush=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate intelligent packing list: {str(e)}"
        )