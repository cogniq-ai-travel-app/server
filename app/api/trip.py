import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from google import genai
from google.genai import types

from app.core.config import settings
from app.models.schemas import PicoResponseSchema, GeneratedCategory
from app.agent.nodes import generate_json_with_fallback

router = APIRouter()
client = genai.Client(
    api_key=settings.GOOGLE_API_KEY,
    http_options=types.HttpOptions(timeout=90000)
)


class TravelerProfile(BaseModel):
    """Once-per-user preferences collected during onboarding."""
    gender: Optional[str] = None
    daily_essentials: List[str] = []
    custom_essentials: List[str] = []


LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt-BR": "Brazilian Portuguese",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "ru": "Russian",
}

COMPANION_LABELS = {
    "solo": "traveling alone",
    "partner": "traveling as a couple",
    "friends": "traveling with friends (everyone packs their own bag)",
    "familyKids": "traveling as a family with kids",
}


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
    notesForPico: Optional[str] = None
    companions: Optional[str] = None
    laundryAccess: Optional[bool] = None
    language: Optional[str] = None
    traveler_profile: Optional[TravelerProfile] = None


class TripCategoriesSchema(BaseModel):
    categories: List[GeneratedCategory]


@router.post("/api/trip/generate")
async def generate_trip_endpoint(request: GenerateTripRequest):
    try:
        print(f"🚀 [TRIP EDITOR] Reviewing and enhancing list for {request.destination}", flush=True)
        
        all_activities = request.activities + request.customActivities

        profile = request.traveler_profile
        traveler_lines = []
        if profile and profile.gender and profile.gender != "undisclosed":
            traveler_lines.append(
                f"Traveler gender: {profile.gender} — tailor wardrobe and toiletry items accordingly."
            )
        if profile and (profile.daily_essentials or profile.custom_essentials):
            essentials = ", ".join(profile.daily_essentials + profile.custom_essentials)
            traveler_lines.append(
                f"Daily essentials the traveler CANNOT forget (include them, priority 'critical'): {essentials}"
            )
        if request.companions:
            traveler_lines.append(
                f"Group: {COMPANION_LABELS.get(request.companions, request.companions)} — adjust shared items and kid essentials to match."
            )
        if request.laundryAccess is not None:
            traveler_lines.append(
                "Laundry access: YES — reduce clothing quantities, the traveler can wash mid-trip."
                if request.laundryAccess
                else "Laundry access: NO — clothing quantities must cover the full trip."
            )
        if request.notesForPico:
            traveler_lines.append(f"Traveler notes: {request.notesForPico}")

        traveler_block = "\n        ".join(traveler_lines) if traveler_lines else "None provided."

        language_name = LANGUAGE_NAMES.get(request.language or "en", "English")

        prompt_text = f"""
        You are Zippy, an expert travel packing assistant.
        My internal rules engine has generated a "Baseline Packing List" for the user.
        Your job is to act as the final Editor. You must read the Baseline List, remove unnecessary items, and add highly specific, localized items based on the trip details.

        --- TRIP DETAILS ---
        Destination: {request.destination} (Coming from {request.fromLocation})
        Duration: {request.durationDays} days ({request.startDate} to {request.endDate})
        Vibe: {request.tripVibe}
        Style: {request.packingStyle}
        Activities: {', '.join(all_activities) if all_activities else 'None specified'}

        --- TRAVELER PROFILE ---
        {traveler_block}

        --- BASELINE PACKING LIST (From Rules Engine) ---
        {json.dumps(request.baseline_categories, indent=2)}

        --- YOUR MISSION ---
        1. KEEP standard items from the Baseline List that make sense. Adjust their 'quantity' to perfectly match the {request.durationDays} day duration.
        2. REMOVE items from the Baseline List that do not fit the specific destination, weather, or 'packingStyle' (e.g., if Style is 'light', aggressively remove 'optional' items).
        3. ADD new, highly specific items that the Rules Engine missed (e.g., if going to a rainy destination, add an umbrella; if going to a specific country, add the correct power adapter type).
        4. SPECIFIC PROFILE RULES:
           - GENDER: Look at the traveler gender under "TRAVELER PROFILE".
             * If gender is male, do NOT include dresses, skirts, makeup, or female-specific hygiene products.
             * If gender is female, do NOT include male-specific wear (like men's briefs/undershirts).
             * If undisclosed or other, default to a practical, gender-neutral unisex wardrobe.
           - LAUNDRY: Look at the laundry access under "TRAVELER PROFILE".
             * If laundry access is YES, assume they will wash clothes and aggressively reduce clothing quantities (e.g., max 6-7 pairs of underwear/socks/shirts even for a 14-day trip).
             * If laundry access is NO, clothing quantities must scale to cover the full duration of the trip (up to a reasonable max limit).
           - DAILY ESSENTIALS: Locate the daily essentials list. Add ALL listed essentials to their logical categories (e.g. Medication/Lenses in Health, Laptop in Electronics). You MUST mark their 'priority' as 'critical'.
           - COMPANIONS: Adjust checklist items based on travel companions. If traveling with kids/family, add kid essentials and child care items. If traveling with friends, keep items personal (do not suggest shared toothpaste/adapters, they pack their own).
        5. Organize the final items into logical categories.

        CRITICAL INSTRUCTION: You MUST process and include all relevant categories (such as Documents, Clothing, Toiletries, Electronics, Health, etc.) in your output. Do NOT skip or omit entire categories from the baseline list unless they are completely irrelevant to the trip. For example, the user always needs Clothing and Toiletries; you must not omit them.

        CRITICAL OUTPUT RULES:
        - Return ONLY a valid JSON object matching the TripCategoriesSchema schema (containing a 'categories' list of categories and items).
        - You must populate the 'categories' array with the final, edited list.
        - FORMATTING STRICT RULE: When generating the 'name' field for an item, provide ONLY the raw item name. Do NOT prefix the string with "name: " or any other label (e.g., output "Deodorant", NOT "name: Deodorant").
        - LANGUAGE RULE: Write every item 'name' and 'notes' in {language_name}. However, category 'name' fields MUST stay in English and match the baseline category names exactly (e.g. "Documents", "Clothing") so the app can merge them.
        """


        final_dict = generate_json_with_fallback(
            contents=prompt_text,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=TripCategoriesSchema,
            ),
        )
        
        if not final_dict or not final_dict.get("categories"):
             raise ValueError("AI failed to generate a valid edited list.")
             
        ai_categories = final_dict.get("categories", [])
        if not isinstance(ai_categories, list):
            ai_categories = []

        # Helper to normalize category names for matching (case-insensitive, strip, singular/plural checks)
        def normalize_cat_name(name: str) -> str:
            n = name.lower().strip()
            if n.endswith("s") and len(n) > 1:
                n = n[:-1]
            return n

        # Map normalized AI category names to AI category dicts
        ai_cat_map = {}
        for cat in ai_categories:
            if isinstance(cat, dict) and "name" in cat:
                ai_cat_map[normalize_cat_name(cat["name"])] = cat

        final_categories = []
        base_cat_names = set()

        # 1. Process each baseline category. Keep baseline if AI omitted it.
        for base_cat in request.baseline_categories:
            if isinstance(base_cat, dict):
                base_name = base_cat.get("name", "")
                base_norm = normalize_cat_name(base_name)
                base_cat_names.add(base_norm)

                if base_norm in ai_cat_map:
                    final_categories.append(ai_cat_map[base_norm])
                else:
                    print(f"⚠️ [TRIP EDITOR] AI omitted baseline category '{base_name}'. Keeping baseline version.", flush=True)
                    final_categories.append(base_cat)

        # 2. Add any new categories generated by AI that weren't in the baseline list
        for ai_cat in ai_categories:
            if isinstance(ai_cat, dict):
                ai_name = ai_cat.get("name", "")
                ai_norm = normalize_cat_name(ai_name)
                if ai_norm not in base_cat_names:
                    final_categories.append(ai_cat)

        merged_draft = request.model_dump()
        merged_draft["categories"] = final_categories

        return {"updated_draft": merged_draft}

    except Exception as e:
        print(f"\n[TRIP GENERATION ERROR]: {str(e)}\n", flush=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate intelligent packing list: {str(e)}"
        )