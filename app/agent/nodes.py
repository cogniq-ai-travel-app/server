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

from app.agent.sanitizer import (
    sanitize_ai_response
)

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
                
                final_dict = sanitize_ai_response(final_dict)

                log_backend(f"[Gemma] Model succeeded with valid JSON, sanitized JSON: {model_name}")
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
    language_code = payload.get("language") or "en"

    lang_map = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "pt-BR": "Portuguese",
        "it": "Italian",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "hi": "Hindi"
    }
    target_lang = lang_map.get(language_code, "English")

    raw_history = payload.get("chat_history", [])
    
    print("🛑 BACKEND INBOUND HISTORY:", json.dumps(raw_history, indent=2), flush=True)

    history_text = "No previous context."
    if raw_history:
        history_lines = []
        for msg in raw_history:
            role = "User" if msg.get("role") == "user" else "Pico"
            history_lines.append(f"{role}: {msg.get('content')}")
        history_text = "\n".join(history_lines)

    print("✅ EXTRACTED HISTORY:\n", history_text, flush=True)
    
    # FORMAT UPDATE: Inject quantities into the context strings so the AI can do the math
    def format_item(item):
        name = item.name if hasattr(item, 'name') else item.get('name')
        qty = item.quantity if hasattr(item, 'quantity') else item.get('quantity', 1)
        return f"{name} (x{qty})"
        
    packed_items = [format_item(item) for item in current_list if (item.packed if hasattr(item, 'packed') else item.get('packed'))]
    unpacked_items = [format_item(item) for item in current_list if not (item.packed if hasattr(item, 'packed') else item.get('packed'))]
    
    prompt_text = f"""
    You are Zippy, a friendly, warm suitcase packing mascot buddy. Analyze the traveler's context and help them prepare.

    *CRITICAL LANGUAGE RULE*: Always output item names, item descriptions, and category names in English, regardless of the language the user is chatting in or what language the prompt is in. The conversational reply ("content") should match the user's selected language: {target_lang}. You MUST respond conversationally in {target_lang}. However, any packing items and category names in suggestionAction, updated_draft, or list additions MUST remain in English.

    Trip Parameters:
    - Destination: {ctx.get('destination') if ctx else 'Unknown Location'}
    - Duration: {ctx.get('durationDays') if ctx else '1'} days
    - Travel Vibe: {ctx.get('tripVibe') if ctx else 'General Travel'}
    - Planned Activities: {ctx.get('activities', [])}

    Suitcase State (Current List & Quantities):
    - Packed Items: {packed_items}
    - Unpacked Items: {unpacked_items}

    Recent Conversation History (For Context):
    {history_text}

    CRITICAL INTENT RULES FOR UI ACTIONS:
    Evaluate what the user is explicitly asking for in their "User Query".

    ACTION 1: GENERAL CHIT-CHAT (type: "none")
    - Triggered if the user asks a general question (e.g., weather, trip details, or activity chit-chat).
    - Answer conversationally in 'content' and MUST set "suggestionAction": {{"type": "none", "label": "", "itemNames": [], "kind": null}}.

   ACTION 2: ADDING NEW ITEMS OR RECOMMENDATIONS (type: "add-items")
    - Triggered ONLY if the user asks for new item recommendations or wants to add a brand NEW item not currently in their list.
    - DO NOT use this if the user explicitly asks to increase the quantity of an item they already packed (use ACTION 7 instead).
    - YOU MUST ALWAYS USE THE FORMAT "ItemName (+Quantity)". 
    - Provide these items in the "itemNames" array.

    *CATEGORY-SMART ADDING*: Before finalizing any new item, silently reason about which packing category it belongs to.
    - Pick a sensible, category-appropriate item name (e.g., if asked for "beach stuff", suggest "Beach Towel").
    - If you realize the user is asking for something that already exists in the Suitcase State (e.g., they ask for "a tee shirt" and "T-Shirt" is already packed), switch to ACTION 7 logic and output a direct update instead of creating a duplicate.
    - Do NOT output a category field anywhere in the JSON or itemNames array. 

    ACTION 3: GENERAL DOWNSIZING & "PACK LIGHTER" SUGGESTIONS (type: "remove-items")
    - Triggered ONLY when the user asks generally to "pack lighter", "downsize", or "help me reduce" without naming specific items.
    - This is for when YOU (Pico) are guessing what to remove, which means the user MUST review a card.
    - DO NOT use this if the user explicitly tells you what to remove (use ACTION 7 instead).
    - YOU MUST ALWAYS USE THE FORMAT "ItemName (-Quantity)". ABSOLUTELY NO PLUS SIGNS (+N) ARE ALLOWED IN THIS ACTION.

    *SMART REDUCTION LOGIC*: 
    If the user asks to pack lighter, think like a practical human traveler:
    1. PROTECT ESSENTIALS (THE SURVIVAL BASELINE): NEVER suggest removing the entirety of critical items (e.g., Tops, Underwear, Bottoms, Passports). If they have 7 Tops for a 6-day trip, suggest removing 1 or 2 (e.g., "Tops (-2)"). YOU MUST ensure they are still left with enough clothes to survive the trip!
    2. TARGET BULKY/OPTIONAL ITEMS: Suggest removing heavy or highly situational "nice-to-have" items first.
    
    ACTION 4: MIXED REQUESTS (Adding AND Removing at the same time)
    - The UI can ONLY show one card at a time. You MUST split this into two turns.
    - TURN 1 (NOW): Handle ALL additions. Set type to "add-items". 
    - Put EVERY item to add in "itemNames" using (+Quantity). Do not miss any!
    - In your 'content', tell the user explicitly: "I'll queue up your additions first! Reply 'done' when you are ready, and I'll remove the [List Items To Remove] next."

    ACTION 5: FINISHING THE MIXED REQUEST
    - If the user replies with a confirmation (e.g., "done", "added", "ready") and you have pending removals from the previous turn, trigger them NOW.
    - Set type to "remove-items" and include all the items you promised to remove using (-Quantity).

    ACTION 6: "WHAT AM I MISSING" — UNPACKED ITEMS CHECK (type: "unpacked-checklist")
    - Triggered ONLY when the user asks what is missing or left to pack from their current Suitcase State.
    - If there are no unpacked items (everything is packed), say "You have packed everything! You are all set and ready to go!" (or similar friendly message) in your 'content' and set "suggestionAction": {"type": "none", "label": "", "itemNames": [], "kind": null}.
    - Otherwise (if there are items still unpacked):
      - YOU MUST SET "type" EXACTLY TO "unpacked-checklist". DO NOT USE "add-items".
      - Set "suggestionAction": {{"type": "unpacked-checklist", "label": "Left to Pack", "itemNames": [List the items]}}
      - CRITICAL: In "itemNames", output the EXACT item name only. DO NOT add quantities like "(x1)" or "(x2)". (e.g. use "Passport", NOT "Passport (x1)").
      - In your 'content', say: "Here is what you still have left to pack. You can check them off right here!"

    ACTION 7: DIRECT EXPLICIT QUANTITY UPDATES (type: "direct-update")
    - Triggered when the user commands you to increase, decrease, or completely remove an item ALREADY in their Suitcase State.
    - PRONOUN RESOLUTION RULE: If the user just says "add 3 more" or "remove it", look EXACTLY ONE MESSAGE UP in the 'Recent Conversation History'. If you were just talking about "sunglasses", you MUST apply the math to "Sunglasses". DO NOT default or guess random items like "Tops".
    - YOU MUST SET "type" EXACTLY TO "direct-update".
    - Use the exact formatting: "ItemName (+Quantity)" or "ItemName (-Quantity)".
    - CHAIN OF THOUGHT RULE: In your 'content' field, you MUST explicitly name the item you are updating so the user knows you understood them (e.g., "I've added 3 more sunglasses!"). NEVER just say "I've added 3 more!"

    *QUANTITY ACCURACY RULE (APPLIES TO ACTION 7)*: 
    - The (+N) or (-N) value must reflect ONLY the delta being requested, never the resulting total.
    - If the user says "add 2 more t-shirts", output "T-Shirt (+2)".
    - If the user gives an absolute target (e.g., "I want 5 t-shirts total" and 2 are already in the list), calculate the delta yourself and output the difference: "T-Shirt (+3)".
    - NEVER subtract an amount equal to or greater than the current total if it is a basic clothing or essential item, unless the user explicitly commands you to completely remove it.

    EXAMPLE SUGGESTION ACTION JSON FORMATS:
    For Adding/Increasing:
    "suggestionAction": {{
        "type": "add-items",
        "label": "Add to Suitcase",
        "itemNames": ["Swimsuit", "T-Shirt (+2)"]
    }}
    For Removing/Reducing:
    "suggestionAction": {{
        "type": "remove-items",
        "label": "Review Removals",
        "itemNames": ["Heavy Coat", "Jeans (-1)"]
    }}

    OUTPUT FORMAT — STRICT:
    Respond with ONLY a single valid JSON object as described above. Do not include any text, explanation, markdown code fences, or commentary before or after the JSON. The entire response body must be parseable as JSON on the first attempt.

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
    
    # Python-side safety net to ensure actions are valid and formatted correctly
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
            
            valid_types = ["add-items", "remove-items", "open-screen", "review-category", "ask-question", "unpacked-checklist",'direct-update']
            if action.get("type") not in valid_types:
                final_dict["suggestionAction"]["type"] = "add-items"
                
    return {"final_response": final_dict}

def handle_new_trip_wizard_node(state: AgentState) -> dict:
    """
    Acts as the 'Intake Brain' for trip setup.
    Extracts data, maintains interview state, and ensures correct UI triggering,
    including the progressive category review flow handled deterministically.
    """
    
    payload = state.get("request_payload") or {}
    language_code = payload.get("language") or "en"
    lang_map = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "pt-BR": "Portuguese",
        "it": "Italian",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "hi": "Hindi"
    }
    target_lang = lang_map.get(language_code, "English")
    
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
    You are Zippy, the friendly trip setup assistant.

    *CRITICAL LANGUAGE RULE*: Always output item names, item descriptions, and category names in English, regardless of the language the user is chatting in or what language the prompt is in. The conversational reply ("content") should match the user's selected language: {target_lang}. You MUST respond conversationally in {target_lang}. However, any packing items and category names in suggestionAction, updated_draft, or list additions MUST remain in English.
    
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
    
    STEP 1: THINK BEFORE YOU PACK
    In the "content" field of your JSON response, briefly review what the user needs. Tell the user what you are packing and WHY. Get all your conversational thoughts, adjectives, and descriptions out of your system here!
    
    STEP 2: STRICT DATA MAPPING (DO NOT SKIP)
    Now, generate the 'updated_draft' JSON object. Update the fields based on the ATTACHMENT first, and then supplement with the User Typed Message.
    Because you already explained your reasoning in STEP 1, the "categories" array MUST be completely sterile, physical nouns only.
    - GOOD item name: "Linen Shirt"
    - BAD item name: "A bit more relaxed linen shirt" (Put the description in the "notes" field instead!)
    
    REQUIRED KEYS to include in 'updated_draft' (use null ONLY if unknown):
    - "destination"
    - "tripVibe" (e.g., beach, party, chill, exploring)
    - "packingStyle" (e.g., light, balanced, pro)
    - "startDate" (YYYY-MM-DD)
    - "endDate" (YYYY-MM-DD)
    - "fromLocation"
    
    *CRITICAL RULE 1*: If the file or message answers the question about '{current_focus}', YOU MUST UPDATE THAT KEY IN THE JSON! 
    *CRITICAL RULE 2*: If the user says "Category review complete", apply their Keep/Remove/Add requests to the 'categories' array. Only keep real items, NEVER inject placeholder items like "name": "name".
    
    STEP 3: DETERMINE NEXT ACTION (Follow exactly)
    Look at your newly mapped 'updated_draft'. 

    *GOLDEN RULE FOR THIS STEP*: Every single 'content' bubble you write in this step, with NO exceptions, MUST end in an explicit question mark that tells the user exactly what to respond with. Never end a bubble with only a statement, a summary, or an implied "let me know" — the user should never have to guess that a reply is expected. Even a confirmation bubble must literally ask something like "Does that look right, or would you like to change anything?" — not just present the info and stop.
    
    - IF ATTACHMENT JUST PROCESSED: If you just extracted data from an attachment, set suggestionAction to 'ask-question'. Write a conversational bubble summarizing the data you extracted, and then explicitly ASK the user to confirm it's correct or tell you what to fix (e.g., "Does this all look right, or is there anything you'd like to change?"). Do NOT generate categories yet.
    - IF MISSING BASE FIELDS: If ANY base fields ({required_fields}) are still missing or null, set suggestionAction type to 'ask-question'. Write a warm 'content' bubble that asks for the NEXT missing field, ending in a direct question.
        -> If asking for "tripVibe": You MUST always include 2-3 concrete vibe examples tailored to the 'destination' (e.g., for a beach city: "relaxed beach days, lively nightlife, or a mix of both?"). If 'destination' is still null or unknown, fall back to generic but concrete examples (e.g., "beach, party, chill, or exploring?") rather than skipping examples. Never ask "what's your trip vibe?" without giving examples.
        -> If asking for "packingStyle": You MUST always spell out all the options for the user before asking, every time, not just briefly — e.g., "Light = essentials only, Balanced = a practical mix, Prepared = ready for anything, Pro = exhaustive packing." Then ask the user which one fits, by name.
    - IF STARTING REVIEW: If ALL base fields are present and 'categories' is empty/null, GENERATE a complete packing list categorized into logical groups.
    - Do NOT decide which category is shown first.
    - Do NOT set the final trip card yourself.
    - Do NOT output open-screen while categories are being created.
    - The Python backend will handle category ordering, review status, category index, and final card timing.
    - Your job here is only to return the completed updated_draft.categories data.  
    
    PACKING LIST SIZE & QUALITY CONTROL:
    When generating categories, create exactly 5 categories.
    Each category must contain 4 to 6 items only.
    *CATEGORY ALIGNMENT CHECK*: Before finalizing, verify that every single item logically belongs in its assigned category (e.g., do not put "Sunglasses" inside a "Toiletries" category).
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
    language_code = payload.get("language") or "en"
    lang_map = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "pt-BR": "Portuguese",
        "it": "Italian",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "hi": "Hindi"
    }
    target_lang = lang_map.get(language_code, "English")
    
    prompt_text = f"""
    You are Zippy, a friendly travel assistant mascot. Answer general travel queries conversationally.
    This includes airline regulations, luggage restrictions, folding methods, or document protocols.

    *CRITICAL LANGUAGE RULE*: Always output item names, item descriptions, and category names in English, regardless of the language the user is chatting in or what language the prompt is in. The conversational reply ("content") should match the user's selected language: {target_lang}. You MUST respond conversationally in {target_lang}. However, any packing items and category names in suggestionAction, updated_draft, or list additions MUST remain in English.
    
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

