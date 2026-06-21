import json
import re
import time
from datetime import date, datetime
from google import genai
from google.genai import types
from app.core.config import settings
from app.models.schemas import PicoResponseSchema
from app.agent.state import AgentState
from typing import Dict, Any, Optional
import base64

client = genai.Client(
    api_key=settings.GOOGLE_API_KEY,
    http_options=types.HttpOptions(timeout=90000)
)


def log_backend(message: str):
    print(message, flush=True)


def parse_ai_json(text_response: str) -> dict:
    """Parse model JSON even when it is wrapped in markdown or extra text."""
    if not text_response:
        raise ValueError("Model returned an empty response.")

    clean_text = text_response.strip()

    if clean_text.startswith("```json"):
        clean_text = clean_text[7:].strip()

    if clean_text.startswith("```"):
        clean_text = clean_text[3:].strip()

    if clean_text.endswith("```"):
        clean_text = clean_text[:-3].strip()

    try:
        parsed = json.loads(clean_text)

        if isinstance(parsed, str):
            parsed = json.loads(parsed)

        if not isinstance(parsed, dict):
            raise ValueError("Parsed model output was not a JSON object.")

        return parsed
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean_text, flags=re.DOTALL)

        if not match:
            raise

        parsed = json.loads(match.group(0))

        if isinstance(parsed, str):
            parsed = json.loads(parsed)

        if not isinstance(parsed, dict):
            raise ValueError("Extracted model output was not a JSON object.")

        return parsed


def response_to_dict(response) -> dict:
    """Prefer structured parsed output, then fall back to text JSON."""
    parsed = getattr(response, "parsed", None)

    if parsed is not None:
      if hasattr(parsed, "model_dump"):
          return parsed.model_dump(exclude_none=True)

      if isinstance(parsed, dict):
          return parsed

    text = getattr(response, "text", None)

    if not text:
        raise ValueError("Model response had no text or parsed payload.")

    return parse_ai_json(text)


def generate_content_with_fallback(*, contents, config):
    """
    Kept for compatibility. This only retries generation errors.
    Prefer generate_json_with_fallback for Pico JSON responses.
    """
    model_names = [
        settings.GEMMA_PRIMARY_MODEL,
        settings.GEMMA_FALLBACK_MODEL,
    ]

    last_error = None
    tried = set()

    for model_name in model_names:
        if not model_name or model_name in tried:
            continue

        tried.add(model_name)

        # Try each model up to 2 times (1 initial try + 1 retry)
        for attempt in range(2):
            try:
                log_backend(f"[Gemma] Trying model: {model_name} (attempt {attempt + 1}/2)")
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                last_error = exc
                log_backend(f"[Gemma] Model attempt failed: {model_name} (attempt {attempt + 1}/2) -> {exc}")
                if attempt == 0:
                    time.sleep(1)

    raise last_error or RuntimeError("No Gemma model was available.")


def generate_json_with_fallback(*, contents, config) -> dict:
    """
    Retries both model-call failures and malformed JSON failures.
    This is what Pico endpoints should use.
    """
    model_names = [
        settings.GEMMA_PRIMARY_MODEL,
        settings.GEMMA_FALLBACK_MODEL,
    ]

    tried = set()
    failures = []

    for model_name in model_names:
        if not model_name or model_name in tried:
            continue

        tried.add(model_name)

        # Try each model up to 2 times (1 initial try + 1 retry)
        for attempt in range(2):
            try:
                started_at = time.time()
                log_backend(f"[Gemma] Trying model: {model_name} (attempt {attempt + 1}/2)")

                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )

                duration_ms = round((time.time() - started_at) * 1000)
                log_backend(f"[Gemma] Raw model response received from {model_name} in {duration_ms}ms (attempt {attempt + 1}/2)")

                final_dict = response_to_dict(response)

                if not isinstance(final_dict, dict):
                    raise ValueError("Model output was not a JSON object.")

                log_backend(f"[Gemma] Model succeeded with valid JSON: {model_name}")
                return final_dict

            except Exception as exc:
                failure_message = f"{model_name} (attempt {attempt + 1}/2): {exc}"
                log_backend(f"[Gemma] Model attempt failed or returned invalid JSON: {failure_message}")
                if attempt == 1:
                    failures.append(f"{model_name}: {exc}")
                else:
                    time.sleep(1)

    raise RuntimeError(
        "All Gemma models failed or returned invalid JSON. "
        + " | ".join(failures)
    )

def clean_item_name(value: Any, fallback: str = "Travel item") -> str:
    if not isinstance(value, str):
        return fallback

    cleaned = value.strip()

    lowered = cleaned.lower()
    for prefix in ["name:", "item:", "object:"]:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break

    if (
    not cleaned
    or cleaned.lower() in {"name", "item", "object", "travel item"}
    or re.fullmatch(r"item\s*\d+", cleaned.lower())
    ):
     return fallback

    return " ".join(cleaned.split())


def normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower().strip() if ch.isalnum())


def parse_list_line(line: str, prefix: str) -> list[str]:
    if not line.lower().startswith(prefix.lower()):
        return []

    raw = line.split(":", 1)[1].strip()

    if not raw or raw.lower() == "none":
        return []

    return [
        clean_item_name(part.strip(), "")
        for part in raw.split(",")
        if clean_item_name(part.strip(), "")
    ]


def parse_category_review_message(message: str) -> Optional[dict]:
    if not message.startswith("Category review complete:"):
        return None

    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return None

    category_name = lines[0].replace("Category review complete:", "").strip()

    keep: list[str] = []
    remove: list[str] = []
    add: list[str] = []

    for line in lines[1:]:
        if line.lower().startswith("keep:"):
            keep = parse_list_line(line, "Keep")
        elif line.lower().startswith("remove:"):
            remove = parse_list_line(line, "Remove")
        elif line.lower().startswith("add:"):
            add = parse_list_line(line, "Add")

    return {
        "category_name": category_name,
        "keep": keep,
        "remove": remove,
        "add": add,
    }


def clean_category_items(items: list[Any]) -> list[dict]:
    cleaned_items = []

    for index, item in enumerate(items):
        if isinstance(item, str):
            name = clean_item_name(item, "")
            if name:
                cleaned_items.append({
                    "name": name,
                    "quantity": 1,
                    "priority": "useful",
                    "notes": None,
                })
            continue

        if hasattr(item, "model_dump"):
          item = item.model_dump()

        if isinstance(item, dict):
          name = clean_item_name(item.get("name"), "")
          if not name:
            continue

        raw_quantity = item.get("quantity")
        quantity = raw_quantity if isinstance(raw_quantity, int) and raw_quantity > 0 else 1

        priority = item.get("priority")
        if priority not in {"critical", "useful", "optional"}:
            priority = "useful"

        cleaned_items.append({
            **item,
            "name": name,
            "quantity": quantity,
            "priority": priority,
        })

    return cleaned_items


def clean_categories(categories: list[Any]) -> list[dict]:
    cleaned_categories = []

    for index, category in enumerate(categories):
        if not isinstance(category, dict):
            continue

        name = clean_item_name(category.get("name"), f"Category {index + 1}")
        items = clean_category_items(category.get("items") or [])

        if not items:
            continue

        cleaned_categories.append({
            **category,
            "name": name,
            "items": items,
        })

    return cleaned_categories


def build_review_action(category: dict, index: int, total: int) -> dict:
    return {
        "type": "review-category",
        "label": f"Review {category.get('name')}",
        "categoryName": category.get("name"),
        "categoryIndex": index + 1,
        "totalCategories": total,
        "items": category.get("items", []),
    }
    
def normalize_iso_date_to_upcoming(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return value

    today = date.today()

    if parsed >= today:
        return parsed.isoformat()

    candidate = parsed.replace(year=today.year)

    if candidate < today:
        candidate = candidate.replace(year=today.year + 1)

    return candidate.isoformat()


def normalize_trip_dates_in_draft(draft: dict) -> dict:
    next_draft = dict(draft)

    start_date = normalize_iso_date_to_upcoming(next_draft.get("startDate"))
    end_date = normalize_iso_date_to_upcoming(next_draft.get("endDate"))

    if isinstance(start_date, str):
        next_draft["startDate"] = start_date

    if isinstance(end_date, str):
        next_draft["endDate"] = end_date

    if (
        isinstance(next_draft.get("startDate"), str)
        and isinstance(next_draft.get("endDate"), str)
        and next_draft["endDate"] < next_draft["startDate"]
    ):
        try:
            end_as_date = datetime.strptime(next_draft["endDate"], "%Y-%m-%d").date()
            next_draft["endDate"] = end_as_date.replace(
                year=end_as_date.year + 1
            ).isoformat()
        except ValueError:
            pass

    return next_draft


def get_review_index(draft: dict) -> int:
    raw_index = draft.get("picoReviewIndex")

    if isinstance(raw_index, int):
        return max(0, raw_index)

    try:
        return max(0, int(raw_index))
    except Exception:
        return 0


def make_review_response(draft: dict, index: int, content: str) -> dict:
    categories = clean_categories(draft.get("categories") or [])

    if not categories:
        return {
            "current_draft": draft,
            "final_response": {
                "content": "I need to rebuild the packing categories before we continue.",
                "suggestionAction": {
                    "type": "ask-question",
                    "label": "Continue planning",
                    "itemNames": [],
                    "kind": None,
                },
                "updated_draft": draft,
            },
        }

    safe_index = min(max(index, 0), len(categories) - 1)
    category = categories[safe_index]

    next_draft = {
        **draft,
        "categories": categories,
        "picoReviewStatus": "reviewing",
        "picoReviewIndex": safe_index,
    }

    return {
        "current_draft": next_draft,
        "final_response": {
            "content": content,
            "suggestionAction": build_review_action(
                category,
                safe_index,
                len(categories),
            ),
            "updated_draft": next_draft,
        },
    }


def make_final_trip_response(draft: dict) -> dict:
    categories = clean_categories(draft.get("categories") or [])

    next_draft = normalize_trip_dates_in_draft({
        **draft,
        "categories": categories,
        "picoReviewStatus": "readyToSave",
        "picoReviewIndex": len(categories),
    })

    destination = next_draft.get("destination") or "your trip"

    return {
        "current_draft": next_draft,
        "final_response": {
            "content": (
                f"I’ve got all your trip details locked in. "
                f"You’re going to have an amazing time exploring {destination}, "
                "and your packing list is ready to save."
            ),
            "suggestionAction": {
                "type": "open-screen",
                "label": "Add to Active Trips",
                "route": "/(tabs)/pack",
                "itemNames": [],
                "kind": None,
            },
            "updated_draft": next_draft,
        },
    }


def default_items_for_category(category_name: str, draft: dict) -> list[dict]:
    destination = draft.get("destination") or "your destination"

    defaults = {
        "Clothing": [
            {"name": "Comfortable outfits", "quantity": 4, "priority": "critical"},
            {"name": "Walking shoes", "quantity": 1, "priority": "critical"},
            {"name": "Light jacket", "quantity": 1, "priority": "useful"},
            {"name": "Sleepwear", "quantity": 2, "priority": "useful"},
        ],
        "Toiletries": [
            {"name": "Toothbrush", "quantity": 1, "priority": "critical"},
            {"name": "Toothpaste", "quantity": 1, "priority": "critical"},
            {"name": "Deodorant", "quantity": 1, "priority": "critical"},
            {"name": "Skincare basics", "quantity": 1, "priority": "useful"},
        ],
        "Electronics": [
            {"name": "Phone charger", "quantity": 1, "priority": "critical"},
            {"name": "Power bank", "quantity": 1, "priority": "critical"},
            {"name": "Universal adapter", "quantity": 1, "priority": "critical"},
            {"name": "Charging cables", "quantity": 2, "priority": "useful"},
        ],
        "Documents": [
            {"name": "Passport", "quantity": 1, "priority": "critical"},
            {"name": "Visa or entry permit", "quantity": 1, "priority": "critical"},
            {"name": "Travel insurance", "quantity": 1, "priority": "critical"},
            {"name": "Flight tickets", "quantity": 1, "priority": "critical"},
        ],
        "Trip Extras": [
            {"name": "Reusable water bottle", "quantity": 1, "priority": "useful"},
            {"name": "Small daypack", "quantity": 1, "priority": "useful"},
            {"name": "Umbrella", "quantity": 1, "priority": "useful"},
            {
                "name": f"{destination} notes",
                "quantity": 1,
                "priority": "optional",
            },
        ],
    }

    return defaults.get(category_name, [])


def supplement_missing_categories(draft: dict, categories: list[dict]) -> list[dict]:
    required_names = [
        "Clothing",
        "Toiletries",
        "Electronics",
        "Documents",
        "Trip Extras",
    ]

    cleaned_categories = clean_categories(categories)
    cleaned_categories = [
    category
    for category in cleaned_categories
    if category.get("items") and len(category.get("items", [])) >= 2
    ]
    existing_names = {
        normalize_key(category.get("name", ""))
        for category in cleaned_categories
    }

    final_categories = list(cleaned_categories)

    for category_name in required_names:
        if len(final_categories) >= 5:
            break

        if normalize_key(category_name) in existing_names:
            continue

        final_categories.append({
            "name": category_name,
            "items": default_items_for_category(category_name, draft),
        })

    return clean_categories(final_categories[:5])


def handle_category_review_without_llm(draft: dict, user_message: str) -> Optional[dict]:
    parsed = parse_category_review_message(user_message)

    if not parsed:
        return None

    categories = clean_categories(draft.get("categories") or [])

    if not categories:
        return None

    category_name = parsed["category_name"]
    category_index = -1

    for index, category in enumerate(categories):
        if category.get("name", "").lower() == category_name.lower():
            category_index = index
            break

    if category_index == -1:
        return make_review_response(
            draft,
            get_review_index(draft),
            "Let’s continue with the category currently on screen.",
        )

    category = categories[category_index]
    keep_keys = {normalize_key(name) for name in parsed["keep"]}
    remove_keys = {normalize_key(name) for name in parsed["remove"]}

    next_items = []

    for item in category.get("items", []):
        item_name = clean_item_name(item.get("name"), "")
        item_key = normalize_key(item_name)

        if not item_name:
            continue

        if keep_keys:
            if item_key in keep_keys:
                next_items.append({**item, "name": item_name})
        elif item_key not in remove_keys:
            next_items.append({**item, "name": item_name})

    for added_name in parsed["add"]:
        clean_name = clean_item_name(added_name, "")
        if not clean_name:
            continue

        next_items.append({
            "name": clean_name,
            "quantity": 1,
            "priority": "useful",
            "notes": "Added during Pico review",
        })

    categories[category_index] = {
        **category,
        "items": next_items,
    }

    merged_draft = normalize_trip_dates_in_draft({
        **draft,
        "categories": categories,
    })

    next_index = category_index + 1

    if next_index < len(categories):
        return make_review_response(
            {
                **merged_draft,
                "picoReviewIndex": next_index,
                "picoReviewStatus": "reviewing",
            },
            next_index,
            f"Saved! Let’s check {categories[next_index].get('name')} next.",
        )

    return make_final_trip_response(merged_draft)

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
    
    final_dict = generate_json_with_fallback(
        contents=prompt_text,
        config=types.GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json",
            response_schema=PicoResponseSchema,
        ),
    )
    
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
    
    draft = normalize_trip_dates_in_draft(
        state.get("current_draft") or payload.get("current_draft") or {}
    )
    attachment = payload.get("attachment")
    user_message = payload.get("user_message", "")

    deterministic_review = handle_category_review_without_llm(draft, user_message)
    if deterministic_review:
       return deterministic_review

    existing_categories = clean_categories(draft.get("categories") or [])
    if existing_categories:
        existing_categories = supplement_missing_categories(draft, existing_categories)

        review_status = draft.get("picoReviewStatus")
        review_index = get_review_index(draft)

        if review_status == "readyToSave" or review_index >= len(existing_categories):
            return make_final_trip_response({
                **draft,
                "categories": existing_categories,
            })

        # If categories already exist, do not let the LLM restart or rewrite the flow.
        # Only continue from the saved review index.
        return make_review_response(
            {
                **draft,
                "categories": existing_categories,
                "picoReviewStatus": "reviewing",
                "picoReviewIndex": review_index,
            },
            review_index,
            f"Let’s continue with {existing_categories[review_index].get('name')}.",
        )
    
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
    today_iso = date.today().isoformat()
    
    prompt_text = f"""
    You are Pico, the friendly trip setup assistant.
    
    Today's date is {today_iso}.
    If the user gives dates without a year, infer the next upcoming valid date range.
    Never output past dates for a new trip.
    
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
    - Do NOT decide which category is shown first.
    - Do NOT set the final trip card yourself.
    - Do NOT output open-screen while categories are being created.
    - The Python backend will handle category ordering, review status, category index, and final card timing.
    - Your job here is only to return the completed updated_draft.categories data.  
    
    PACKING LIST SIZE CONTROL:
    When generating categories, create exactly 5 categories.
    Each category must contain 4 to 6 items only.
    Keep item names short and practical.
    Do not write long notes unless truly necessary.
    """
    
    if attachment and attachment.get("base64_data"):
        contents_list.append("The user has uploaded a file. Please read the full text/content of this file and use it to populate the required fields in the 'updated_draft' JSON.")
    
    contents_list.append(prompt_text)
    
    final_dict = generate_json_with_fallback(
    contents=contents_list,
    config=types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=PicoResponseSchema,
    ),
)

    raw_ai_update = final_dict.get("updated_draft") or {}
    ai_updated_draft = {key: value for key, value in raw_ai_update.items() if value is not None}
    
    merged_draft = {**draft, **ai_updated_draft} 
    still_missing = [field for field in required_fields if not merged_draft.get(field)]
    
    action = final_dict.get("suggestionAction") or {}
    categories = merged_draft.get("categories") or []
    categories = clean_categories(categories)
    if categories:
        merged_draft["categories"] = categories

    if still_missing:
        true_next_target = still_missing[0]
        is_invalid_action = action.get("type") in ["open-screen", "review-category", "none"]

        ai_skipped_saving = (
            action.get("type") == "ask-question"
            and current_focus in still_missing
            and user_message.strip()
        )

        is_attachment_confirmation = (
            attachment is not None and action.get("type") == "ask-question"
        )

        if (is_invalid_action or ai_skipped_saving) and not is_attachment_confirmation:
            print(
                f"⚠️ [GUARDRAIL] Intercepted LLM. Missing: {still_missing}. "
                f"Forcing target: {true_next_target}",
                flush=True,
            )

            final_dict["suggestionAction"] = {
                "type": "ask-question",
                "label": f"Provide {true_next_target}",
                "itemNames": [],
                "kind": None,
            }

            target_clean = true_next_target.replace("trip", "")
            final_dict["content"] = (
                f"Just to make sure I have it right, what is your {target_clean}?"
            )

            final_dict["updated_draft"] = merged_draft

            return {
                "current_draft": merged_draft,
                "final_response": final_dict,
            }

    # Important: this must be OUTSIDE the still_missing block.
    # If all required trip fields are present and categories exist,
    # start the deterministic one-by-one category review flow.
    if categories:
        categories = supplement_missing_categories(merged_draft, categories)

        merged_draft = normalize_trip_dates_in_draft({
            **merged_draft,
            "categories": categories,
            "picoReviewStatus": "reviewing",
            "picoReviewIndex": 0,
        })

        return make_review_response(
            merged_draft,
            0,
            (
                "I’ve got all the trip details I need. "
                f"I’ve built your packing list, starting with {categories[0].get('name')}."
            ),
        )

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
    
    final_dict = generate_json_with_fallback(
        contents=prompt_text,
        config=types.GenerateContentConfig(
            temperature=0.6,
            response_mime_type="application/json",
            response_schema=PicoResponseSchema,
        ),
    )
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

