import json
from google import genai
from google.genai import types
from app.core.config import settings
from app.models.schemas import PicoResponseSchema
from app.agent.state import AgentState
from typing import Dict, Any, Optional
import base64

client = genai.Client(api_key=settings.GOOGLE_API_KEY)

def parse_ai_json(text_response: str) -> dict:
    """Safely strips markdown formatting blocks from LLM JSON outputs."""
    clean_text = text_response.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
        
    return json.loads(clean_text.strip())

def handle_active_trip_node(state: AgentState) -> dict:
    """
    Handles deep, context-aware analysis for an existing physical packing checklist.
    Cross-references packed vs unpacked statuses to guide the user intelligently.
    """
    payload = state["request_payload"]
    
    ctx = payload.get("trip_context") or {}
    current_list = payload.get("current_list", [])
    user_message = payload.get("user_message", "")
    
    packed_items = [item.name if hasattr(item, 'name') else item.get('name') for item in current_list if (item.packed if hasattr(item, 'packed') else item.get('packed'))]
    unpacked_items = [item.name if hasattr(item, 'name') else item.get('name') for item in current_list if not (item.packed if hasattr(item, 'packed') else item.get('packed'))]
    
    prompt_text = f"""
    You are Pico, a friendly, warm suitcase packing mascot buddy.
    Analyze the traveler's context and help them prepare. 
    
    Trip Parameters:
    - Destination: {ctx.get('destination') if ctx else 'Unknown Location'}
    - Duration: {ctx.get('durationDays') if ctx else '1'} days
    - Travel Vibe: {ctx.get('tripVibe') if ctx else 'General Travel'}
    - Planned Activities: {ctx.get('activities', [])}
    
    Suitcase State:
    - Packed Items (Do Not Suggest Adding These!): {packed_items}
    - Unpacked Items: {unpacked_items}
    
    CRITICAL INTENT RULES FOR UI ACTIONS:
    - Evaluate what the user is explicitly asking for in their "User Query".
    - If the user is simply asking a general or contextual question about the trip (e.g., "Tell me about the trip", "What are my trip details?", weather, vibe descriptions, or activity chit-chat), you MUST answer conversationally in 'content' and strictly set "suggestionAction": {{"type": "none", "label": "", "itemNames": [], "kind": null}}.
    - ONLY provide an active recommendation action block if the user explicitly asks to modify or check items (e.g., "What am I missing?", "Suggest for activities", "Help me pack lighter", "What should I remove?").
    
    IF AND ONLY IF PACKING MODIFICATIONS ARE REQUESTED:
    - If suggesting new items to pack, you MUST set "type": "add-items". Do NOT suggest items that are already in Packed Items.
    - If the user asks to pack lighter or remove items, you ABSOLUTELY MUST set "type": "remove-items". Do NOT use 'none'.
    
    EXAMPLE REMOVE JSON FORMAT:
    "suggestionAction": {{
        "type": "remove-items",
        "label": "Remove Heavy Gear",
        "itemNames": ["jeans", "boots"]
    }}
    
    User Query: {user_message}
    """
    
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt_text,
        config=types.GenerateContentConfig(
            temperature=0.7, 
            response_mime_type="application/json",
            response_schema=PicoResponseSchema 
        )
    )
    
    final_dict = parse_ai_json(response.text)
    action = final_dict.get("suggestionAction")
    
    if action and action.get("type") != "none":
        item_names = action.get("itemNames", [])
        
        if not item_names or len(item_names) == 0:
            final_dict["suggestionAction"] = {
                "type": "none",
                "label": "",
                "itemNames": [],
                "kind": None
            }
        else:
            user_msg = user_message.lower()
            ai_label = action.get("label", "").lower()
            
            if "remove" in user_msg or "lighter" in user_msg or "remove" in ai_label:
                final_dict["suggestionAction"]["type"] = "remove-items"
            elif action.get("type") not in ["add-items", "remove-items"]:
                final_dict["suggestionAction"]["type"] = "add-items"
    
    return {"final_response": final_dict}

def handle_new_trip_wizard_node(state: AgentState) -> dict:
    """
    Acts as the 'Intake Brain' for trip setup.
    Extracts data, maintains interview state, and ensures correct UI triggering,
    including the progressive category review flow handled deterministically.
    """
    
    payload = state.get("request_payload") or {}
    
    draft = state.get("current_draft") or payload.get("current_draft") or {}
    attachment = payload.get("attachment")
    user_message = payload.get("user_message", "")
    
    required_fields = ["destination", "tripVibe", "packingStyle", "startDate", "endDate", "fromLocation"]
    missing_fields = [field for field in required_fields if not draft.get(field)]

    contents_list = []

    if attachment and attachment.get("base64_data"):
        try:
            raw_bytes = base64.b64decode(attachment["base64_data"])
            mime_type = attachment.get("mime_type", "application/pdf")
            
            contents_list.append(
                types.Part.from_bytes(
                    data=raw_bytes,
                    mime_type=mime_type
                )
            )
            print(f"[Backend] Successfully processed attached asset of type: {mime_type}")
        except Exception as err:
            print(f"[Backend Error] Failed parsing attachment stream: {err}")

    current_focus = missing_fields[0] if missing_fields else "Category Generation & Review"
    
    prompt_text = f"""
    You are Pico, the friendly trip setup assistant.
    
    Current Trip Draft Memory: {json.dumps(draft)}
    What we still need: {missing_fields}
    The field we just asked the user for: {current_focus}
    
    User Typed Message: "{user_message}"
    ATTACHMENT STATUS: {"[FILE ATTACHED - EXTRACT ALL TEXT DATA]" if attachment else "NO FILE ATTACHED"}
    
    CRITICAL INSTRUCTIONS (EXECUTE STRICTLY IN ORDER):
    
    STEP 0: ATTACHMENT PROCESSING
    If a file is attached, YOU MUST parse its content first. Extract the destination, dates, trip vibe, packing style, and origin location from the file content. 
    Do not ask the user for information that is already present in the attached file.
    
    STEP 1: DATA MAPPING (DO NOT SKIP)
    You MUST output a fully populated 'updated_draft' JSON object. Update the fields based on the ATTACHMENT first, and then supplement with the User Typed Message.
    
    REQUIRED KEYS to include in 'updated_draft' (use null ONLY if unknown):
    - "destination"
    - "tripVibe" (e.g., beach, party, chill, exploring)
    - "packingStyle" (e.g., light, balanced, pro)
    - "startDate" (YYYY-MM-DD)
    - "endDate" (YYYY-MM-DD)
    - "fromLocation"
    
    *CRITICAL RULE 1*: If the file or message answers the question about '{current_focus}', YOU MUST UPDATE THAT KEY IN THE JSON! 
    *CRITICAL RULE 2*: If the user says "Category review complete", apply their Keep/Remove/Add requests to the 'categories' array. Only keep real items, NEVER inject placeholder items like "name": "name".
    
    STEP 2: DETERMINE NEXT ACTION (Follow exactly)
    Look at your newly mapped 'updated_draft'. 
    
    - IF ATTACHMENT JUST PROCESSED: If you just extracted data from an attachment, set suggestionAction to 'ask-question'. Write a conversational bubble summarizing the data you extracted and ask if it looks correct before moving forward. Do NOT generate categories yet.
    - IF MISSING BASE FIELDS: If ANY base fields ({required_fields}) are still missing or null, set suggestionAction type to 'ask-question'. Write a warm 'content' bubble that asks for the NEXT missing field.
    - IF STARTING REVIEW: If ALL base fields are present and 'categories' is empty/null, GENERATE a complete packing list categorized into logical groups. 
    - Note: The Python backend will handle the category indexing. Just output the data.
    """
    
    if attachment and attachment.get("base64_data"):
        contents_list.append("The user has uploaded a file. Please read the full text/content of this file and use it to populate the required fields in the 'updated_draft' JSON.")
    
    contents_list.append(prompt_text)
    
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=contents_list,
        config=types.GenerateContentConfig(
            temperature=0.1, 
            response_mime_type="application/json",
            response_schema=PicoResponseSchema
        )
    )
    
    
    final_dict = parse_ai_json(response.text)

    raw_ai_update = final_dict.get("updated_draft") or {}
    ai_updated_draft = {key: value for key, value in raw_ai_update.items() if value is not None}
    
    merged_draft = {**draft, **ai_updated_draft} 
    still_missing = [field for field in required_fields if not merged_draft.get(field)]
    
    action = final_dict.get("suggestionAction") or {}
    categories = merged_draft.get("categories") or []

    if still_missing:
        true_next_target = still_missing[0]
        is_invalid_action = action.get("type") in ["open-screen", "review-category", "none"]
        
        ai_skipped_saving = action.get("type") == "ask-question" and current_focus in still_missing and user_message.strip()
        
        is_attachment_confirmation = attachment is not None and action.get("type") == "ask-question"

        if (is_invalid_action or ai_skipped_saving) and not is_attachment_confirmation:
            print(f"⚠️ [GUARDRAIL] Intercepted LLM. Missing: {still_missing}. Forcing target: {true_next_target}")
            
            final_dict["suggestionAction"] = {
                "type": "ask-question",
                "label": f"Provide {true_next_target}",
                "itemNames": [],
                "kind": None
            }
            
            target_clean = true_next_target.replace("trip", "")
            final_dict["content"] = f"Just to make sure my systems have it locked in perfectly, what is your {target_clean}?"

    elif categories:

        if not draft.get("categories"):
            first_cat = categories[0]
            final_dict["suggestionAction"] = {
                "type": "review-category",
                "label": f"Review {first_cat.get('name')}",
                "categoryName": first_cat.get("name"),
                "categoryIndex": 1,
                "totalCategories": len(categories),
                "items": first_cat.get("items", [])
            }
            final_dict["content"] = f"Awesome! I've put together your list. Let's review your items starting with {first_cat.get('name')}."

        else:
            current_cat_name = None
            
            if "Category review complete:" in user_message:
                first_line = user_message.split("\n")[0]
                current_cat_name = first_line.replace("Category review complete:", "").strip()
                
            if not current_cat_name:
                current_cat_name = action.get("categoryName")

            if current_cat_name:
                current_index = -1
                for idx, cat in enumerate(categories):
                    if cat.get("name", "").lower() == current_cat_name.lower():
                        current_index = idx
                        break
                
                next_index = current_index + 1
                if current_index != -1:
                    if next_index < len(categories):
                        
                        next_cat = categories[next_index]
                        final_dict["suggestionAction"] = {
                            "type": "review-category",
                            "label": f"Review {next_cat.get('name')}",
                            "categoryName": next_cat.get("name"),
                            "categoryIndex": next_index + 1,
                            "totalCategories": len(categories),
                            "items": next_cat.get("items", [])
                        }
                        final_dict["content"] = f"Saved! Let's check the next category: {next_cat.get('name')}."
                    else:
                        final_dict["suggestionAction"] = {
                            "type": "open-screen",
                            "label": "View My Trip",
                            "itemNames": [],
                            "kind": None
                        }
                        final_dict["content"] = "All done! Your packing list is perfectly curated and ready."

    final_dict["updated_draft"] = merged_draft

    return {
        "current_draft": merged_draft, 
        "final_response": final_dict
    }

def handle_general_travel_node(state: AgentState) -> dict:
    """
    Handles global travel rules, luggage guidelines, airline queries, or basic packing tactics.
    Executed when no active trip contexts are present.
    """
    payload = state["request_payload"]
    user_message = payload.get("user_message", "") 
    
    prompt_text = f"""
    You are Pico, a friendly travel assistant mascot. Answer general travel queries conversationally.
    This includes airline regulations, luggage restrictions, folding methods, or document protocols.
    
    CRITICAL RULES FOR UI ACTIONS:
    - Since there is no specific trip attached, reply helpfully and keep suggestionAction type to 'none' by default.
    - ONLY change "type" to "add-items" if the user explicitly names a list of specific things they want to group into an applicable UI checklist card (e.g., "What are the essential items for carry-on?").
    
    User Message: {user_message}
    """
    
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt_text,
        config=types.GenerateContentConfig(
            temperature=0.6, 
            response_mime_type="application/json",
            response_schema=PicoResponseSchema
        )
    )
    
    final_dict = parse_ai_json(response.text)
    action = final_dict.get("suggestionAction")
    
    if action and action.get("type") != "none":
        item_names = action.get("itemNames", [])
        if not item_names or len(item_names) == 0:
            final_dict["suggestionAction"] = {
                "type": "none",
                "label": "",
                "itemNames": [],
                "kind": None
            }
            
    return {"final_response": final_dict}

