from flask import Blueprint, current_app, jsonify, request, render_template, redirect, url_for
from flask_login import current_user
from bson import ObjectId
from datetime import datetime, timedelta, timezone
from io import BytesIO
import os
import json
import re
import random
import time
import uuid
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import requests
from PIL import Image
import google.generativeai as genai
from google.api_core import exceptions as google_api_exceptions
from pymongo.errors import DuplicateKeyError
from werkzeug.local import LocalProxy

from database import get_db
from diet_config import compute_macro_adherence_10pt
from usage_tracker import (
    GUEST_V3_TRIAL_LIMIT,
    guest_v3_trial_status,
    refund_guest_v3_trial,
    try_reserve_guest_v3_trial,
)


v3_bp = Blueprint("v3", __name__)
db = LocalProxy(get_db)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash-lite")
_v3_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        _v3_model = genai.GenerativeModel(GEMINI_MODEL_ID)
    except Exception:
        _v3_model = None


def _gemini_generate(content, max_output_tokens=4096, temperature=0.2, max_retries=4):
    """Call Gemini with retries for transient failures (rate limits, 5xx, empty text).

    Returns:
        ``(response, None)`` on success with non-empty model text.
        ``(None, "rate_limit")`` when quota / per-minute limits are exceeded after retries.
        ``(None, "generation_failed")`` for other failures or empty/blocked responses.
    """
    if not _v3_model:
        return None, "generation_failed"

    retryable_types = (
        google_api_exceptions.ResourceExhausted,
        google_api_exceptions.ServiceUnavailable,
        google_api_exceptions.DeadlineExceeded,
        google_api_exceptions.InternalServerError,
    )

    last_issue = None
    for attempt in range(max_retries):
        try:
            resp = _v3_model.generate_content(content)
            try:
                text = (getattr(resp, "text", None) or "").strip()
            except Exception:
                text = ""
            if text:
                return resp, None
            last_issue = "empty_or_blocked_response"
        except retryable_types as exc:
            last_issue = exc
        except Exception as exc:
            last_issue = exc

        if attempt < max_retries - 1:
            delay = min(8.0, 0.7 * (2**attempt)) + random.random() * 0.4
            time.sleep(delay)

    print(f"[Gemini] generate_content failed after {max_retries} attempt(s): {last_issue!r}")
    if isinstance(last_issue, google_api_exceptions.ResourceExhausted):
        return None, "rate_limit"
    return None, "generation_failed"


def _is_authed():
    return bool(current_user and getattr(current_user, "is_authenticated", False))


def _auth_guard():
    if not _is_authed():
        return None, (jsonify({"success": False, "error": "auth_required"}), 401)
    user_id = ObjectId(current_user.id)

    try:
        migrate_user_history_to_meal_logs(user_id)
    except Exception:
        pass

    return user_id, None


def _guest_uuid_from_cookie():
    """Must use the same signing key as ``app.py`` / ``auth`` (``current_app.secret_key``)."""
    from itsdangerous import URLSafeSerializer

    cookie = request.cookies.get("guest_session")
    if not cookie:
        return None
    try:
        ser = URLSafeSerializer(current_app.secret_key, salt="guest-session")
        return ser.loads(cookie)
    except Exception:
        return None


def _actor_resolve():
    """Return (user_id ObjectId | None, guest_uuid str | None, error_response | None)."""
    if _is_authed():
        user_id = ObjectId(current_user.id)
        try:
            migrate_user_history_to_meal_logs(user_id)
        except Exception:
            pass
        return user_id, None, None
    gid = _guest_uuid_from_cookie()
    if not gid:
        return None, None, (
            jsonify(
                {
                    "success": False,
                    "error": "auth_required",
                    "message": "Reload the page to start a guest session, or sign in.",
                }
            ),
            401,
        )
    return None, gid, None


def _guest_default_context():
    """Minimal context when no user profile exists (anonymous trial)."""
    return {
        "diet_type": "standard_american",
        "allergies": [],
        "food_restrictions": [],
        "meal_timing_preference": None,
        "prep_time_limit": None,
        "budget_per_meal": None,
        "meal_prep_preference": None,
        "class_schedule": {},
        "cooking_skill": None,
        "living_situation": None,
        "goal_type": "maintain_weight",
        "target_weight_kg": None,
        "timeline_weeks": None,
        "secondary_goals": [],
        "daily_calories": 2000,
        "target_protein_g": None,
        "target_carbs_g": None,
        "target_fat_g": None,
        "target_fiber_g": None,
        "target_sodium_mg": None,
        "target_sugar_g": None,
        "height_cm": None,
        "weight_kg": None,
        "age": None,
        "biological_sex": None,
        "activity_level": None,
        "health_conditions": [],
        "medications": None,
        "supplements": [],
    }


def _serialize_oid(doc):
    if not doc:
        return doc
    out = dict(doc)
    if "_id" in out:
        out["_id"] = str(out["_id"])
    if "user_id" in out and out["user_id"] is not None:
        out["user_id"] = str(out["user_id"])
    if "created_by" in out and out["created_by"] is not None:
        out["created_by"] = str(out["created_by"])
    if "challenge_id" in out and out["challenge_id"] is not None:
        out["challenge_id"] = str(out["challenge_id"])
    if "legacy_analysis_id" in out and out["legacy_analysis_id"] is not None:
        out["legacy_analysis_id"] = str(out["legacy_analysis_id"])
    return out


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return int(default)


def _normalize_token(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _meal_name_from_structured(structured, fallback="Meal"):
    if not isinstance(structured, dict):
        return fallback
    mid = structured.get("meal_identification")
    if isinstance(mid, str) and mid.strip():
        return mid.strip()
    if isinstance(mid, dict) and mid.get("name"):
        return str(mid.get("name")).strip()
    if structured.get("meal_name"):
        return str(structured.get("meal_name")).strip()
    return fallback


def _extract_macros(structured):
    structured = structured or {}
    tn = structured.get("total_nutrition") or {}
    return {
        "calories_kcal": _safe_float(structured.get("calories_kcal", tn.get("calories", 0))),
        "protein_g": _safe_float(structured.get("protein_g", tn.get("protein", 0))),
        "carbs_g": _safe_float(structured.get("carbs_g", tn.get("carbs", 0))),
        "fat_g": _safe_float(structured.get("fat_g", tn.get("fat", 0))),
        "fiber_g": _safe_float(structured.get("fiber_g", tn.get("fiber", 0))),
        "sodium_mg": _safe_float(structured.get("sodium_mg", tn.get("sodium", 0))),
    }


def _recipe_matches_context(recipe, context):
    recipe = recipe or {}
    context = context or {}

    diet_type = _normalize_token(context.get("diet_type"))
    recipe_tags = [_normalize_token(t) for t in (recipe.get("diet_tags") or [])]
    if recipe_tags and diet_type and diet_type not in recipe_tags:
        return False

    prep_limit = context.get("prep_time_limit")
    if prep_limit and recipe.get("cook_time_min"):
        if _safe_int(recipe.get("cook_time_min"), 0) > _safe_int(prep_limit, 0):
            return False

    budget_per_meal = context.get("budget_per_meal")
    if budget_per_meal and recipe.get("cost_per_serving"):
        if _safe_float(recipe.get("cost_per_serving"), 0) > _safe_float(budget_per_meal, 0):
            return False

    ingredients_text = " ".join([str(x).lower() for x in (recipe.get("ingredients") or [])])

    for allergy in context.get("allergies", []) or []:
        token = _normalize_token(allergy)
        if not token:
            continue
        if token.replace("_", " ") in ingredients_text or token in ingredients_text:
            return False

    restrictions = set([_normalize_token(x) for x in (context.get("food_restrictions") or [])])
    if "no_pork" in restrictions and any(k in ingredients_text for k in ["pork", "bacon", "ham"]):
        return False
    if "no_beef" in restrictions and "beef" in ingredients_text:
        return False
    if "no_alcohol" in restrictions and any(k in ingredients_text for k in ["wine", "beer", "rum", "vodka", "whiskey", "alcohol"]):
        return False

    return True


def _parse_json_from_text(text):
    if not text:
        return {}
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
    return {}


def _image_from_request(file_storage=None, image_url=None):
    if file_storage and file_storage.filename:
        img = Image.open(file_storage.stream)
        if img.mode not in ["RGB", "L"]:
            img = img.convert("RGB")
        return img
    if image_url:
        clean_url = str(image_url).strip()
        if not clean_url:
            return None
        parsed = urlsplit(clean_url)
        origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else None
        base_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        attempts = [dict(base_headers)]
        if origin:
            hdr = dict(base_headers)
            hdr["Referer"] = origin + "/"
            attempts.append(hdr)

        last_exc = None
        for headers in attempts:
            try:
                resp = requests.get(clean_url, timeout=20, headers=headers)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
                if img.mode not in ["RGB", "L"]:
                    img = img.convert("RGB")
                return img
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc
    return None


def _to_base64_jpeg(img, max_size=640):
    if img is None:
        return None
    import base64

    copy = img.copy()
    copy.thumbnail((max_size, max_size))
    buf = BytesIO()
    copy.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_OPENFOODFACTS_HEADERS = {
    "User-Agent": "Alimento/1.0 (https://world.openfoodfacts.org/data)",
    "Accept": "application/json",
}


def _barcode_candidates(stripped_digits: str):
    """Try UPC-A (12) as EAN-13 (leading 0) and the reverse — OFF often keys products one way."""
    c = re.sub(r"\D", "", stripped_digits or "")
    if not c:
        return []
    out, seen = [], set()

    def push(x):
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    push(c)
    if len(c) == 12:
        push("0" + c)
    if len(c) == 13 and c.startswith("0"):
        push(c[1:])
    return out


def _lookup_barcode_openfoodfacts_once(code):
    try:
        url = f"https://world.openfoodfacts.org/api/v2/product/{code}.json"
        resp = requests.get(url, timeout=18, headers=_OPENFOODFACTS_HEADERS)
        resp.raise_for_status()
        payload = resp.json() or {}
        if payload.get("status") != 1:
            return None
        product = payload.get("product") or {}
        nutr = product.get("nutriments") or {}
        serving_size = product.get("serving_size")
        return {
            "barcode": code,
            "name": product.get("product_name") or product.get("generic_name") or "Unknown packaged food",
            "brand": product.get("brands", ""),
            "serving_size": serving_size,
            "calories_kcal": _safe_float(nutr.get("energy-kcal_serving") or nutr.get("energy-kcal_100g") or nutr.get("energy-kcal")),
            "protein_g": _safe_float(nutr.get("proteins_serving") or nutr.get("proteins_100g") or nutr.get("proteins")),
            "carbs_g": _safe_float(nutr.get("carbohydrates_serving") or nutr.get("carbohydrates_100g") or nutr.get("carbohydrates")),
            "fat_g": _safe_float(nutr.get("fat_serving") or nutr.get("fat_100g") or nutr.get("fat")),
            "fiber_g": _safe_float(nutr.get("fiber_serving") or nutr.get("fiber_100g") or nutr.get("fiber")),
            "sodium_mg": round(_safe_float(nutr.get("sodium_serving") or nutr.get("sodium_100g") or nutr.get("sodium")) * 1000, 2),
            "source": "openfoodfacts",
            "raw": product,
        }
    except Exception:
        return None


def _lookup_barcode_openfoodfacts(barcode):
    """Resolve barcode via Open Food Facts; tries UPC/EAN variants (OFF etiquette: User-Agent)."""
    primary = re.sub(r"\D", "", barcode or "")
    for cand in _barcode_candidates(primary):
        hit = _lookup_barcode_openfoodfacts_once(cand)
        if hit:
            hit["barcode"] = primary
            hit["off_lookup_code"] = cand
            return hit
    return None


def _ai_structured_from_text(meal_text, user_context=None):
    if not meal_text:
        return {}
    if not _v3_model:
        return None

    ctx = user_context or {}
    prompt = f"""
You are a nutrition parser.
Return ONLY valid JSON with keys:
meal_name, calories_kcal, protein_g, carbs_g, fat_g, fiber_g, sodium_mg, notes.

Constraints:
- Numeric fields are numbers.
- notes is one short sentence.

User context:
diet_type={ctx.get('diet_type', 'standard_american')}, allergies={ctx.get('allergies', [])}

Meal text: {meal_text}
"""
    res, _ = _gemini_generate(prompt)
    parsed = _parse_json_from_text(getattr(res, "text", "") if res else "")
    if not parsed:
        return None
    parsed = dict(parsed)

    core = [parsed.get("calories_kcal"), parsed.get("protein_g"), parsed.get("carbs_g"), parsed.get("fat_g")]
    if all(v in [None, "", 0, 0.0] for v in core):
        return None

    return {
        "meal_name": str(parsed.get("meal_name") or meal_text[:60]),
        "calories_kcal": _safe_float(parsed.get("calories_kcal")),
        "protein_g": _safe_float(parsed.get("protein_g")),
        "carbs_g": _safe_float(parsed.get("carbs_g")),
        "fat_g": _safe_float(parsed.get("fat_g")),
        "fiber_g": _safe_float(parsed.get("fiber_g")),
        "sodium_mg": _safe_float(parsed.get("sodium_mg")),
        "notes": str(parsed.get("notes") or "AI generated from meal text."),
    }


def _ai_structured_from_image(img, user_context=None):
    if not img:
        return {}
    if not _v3_model:
        return None

    ctx = user_context or {}
    prompt = f"""
Analyze this meal image and return ONLY JSON.
Keys: meal_name, calories_kcal, protein_g, carbs_g, fat_g, fiber_g, sodium_mg, notes.
Use concise values and realistic estimates.
Diet context: {ctx.get('diet_type', 'standard_american')}
"""
    res, _ = _gemini_generate([prompt, img])
    parsed = _parse_json_from_text(getattr(res, "text", "") if res else "")
    if not parsed:
        return None
    parsed = dict(parsed)

    core = [parsed.get("calories_kcal"), parsed.get("protein_g"), parsed.get("carbs_g"), parsed.get("fat_g")]
    if all(v in [None, "", 0, 0.0] for v in core):
        return None

    return {
        "meal_name": str(parsed.get("meal_name") or "Meal from image"),
        "calories_kcal": _safe_float(parsed.get("calories_kcal")),
        "protein_g": _safe_float(parsed.get("protein_g")),
        "carbs_g": _safe_float(parsed.get("carbs_g")),
        "fat_g": _safe_float(parsed.get("fat_g")),
        "fiber_g": _safe_float(parsed.get("fiber_g")),
        "sodium_mg": _safe_float(parsed.get("sodium_mg")),
        "notes": str(parsed.get("notes") or "AI generated from image."),
    }


def _save_meal_log(payload, *, user_id=None, guest_session_id=None):
    if (user_id is None) == (guest_session_id is None):
        raise ValueError("Exactly one of user_id or guest_session_id must be set")
    now = datetime.now(timezone.utc)
    meal = {
        "schema_version": 3,
        "user_id": user_id,
        "guest_session_id": guest_session_id,
        "source": payload.get("source", "manual"),
        "meal_name": payload.get("meal_name", "Meal"),
        "notes": payload.get("notes", ""),
        "diet_type": payload.get("diet_type") or "standard_american",
        "meal_type": payload.get("meal_type") or "unspecified",
        "macros": {
            "calories_kcal": _safe_float(payload.get("calories_kcal")),
            "protein_g": _safe_float(payload.get("protein_g")),
            "carbs_g": _safe_float(payload.get("carbs_g")),
            "fat_g": _safe_float(payload.get("fat_g")),
            "fiber_g": _safe_float(payload.get("fiber_g")),
            "sodium_mg": _safe_float(payload.get("sodium_mg")),
        },
        "image_base64": payload.get("image_base64"),
        "barcode": payload.get("barcode"),
        "raw_input": payload.get("raw_input"),
        "metadata": payload.get("metadata", {}),
        "personalization": {
            "macro_adherence": payload.get("macro_adherence") or {}
        },
        "logged_at": payload.get("logged_at") or now,
        "created_at": now,
        "updated_at": now,
    }
    result = db.meal_logs.insert_one(meal)
    meal["_id"] = result.inserted_id
    return _serialize_oid(meal)


def _week_start(dt_local):
    d = dt_local.date()
    monday = d - timedelta(days=d.weekday())
    return datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)


def _day_bounds_in_utc(day_str, tz_name):
    tz = ZoneInfo(tz_name or "UTC")
    if day_str:
        try:
            local_day = datetime.strptime(day_str, "%Y-%m-%d").date()
        except Exception:
            local_day = datetime.now(tz).date()
    else:
        local_day = datetime.now(tz).date()

    local_start = datetime(local_day.year, local_day.month, local_day.day, 0, 0, 0, tzinfo=tz)
    local_end = local_start + timedelta(days=1)
    return local_start.astimezone(timezone.utc), local_end.astimezone(timezone.utc), local_day


def _compute_grocery_from_plan(plan_doc):
    stop_exact = {
        "water",
        "ice cubes",
        "optional",
        "to taste",
        "if desired",
        "if any",
        "herbs",
        "spices",
        "seasoning",
        "e.g",
        "eg",
        "i.e",
        "ie",
        "low sodium",
        "for extra protein",
        "prepare according to package directions",
        "according to package directions",
    }

    bad_phrases = [
        "leftover",
        "from feb",
        "from mar",
        "from apr",
        "from may",
        "from jun",
        "from jul",
        "from aug",
        "from sep",
        "from oct",
        "from nov",
        "from dec",
        "according to package directions",
        "for extra protein",
    ]

    def _clean_token(token):
        t = str(token or "").strip().lower()
        if not t:
            return ""

        if any(p in t for p in bad_phrases):
            return ""

        # Remove trailing hints and formatting noise.
        t = re.sub(r"\b(optional|to taste|if desired|if any)\b", "", t)
        t = re.sub(r"\be\.?g\.?\b", "", t)
        t = re.sub(r"\bi\.?e\.?\b", "", t)
        t = t.replace("&", " and ")

        # Remove leading connector words and generic descriptors.
        t = re.sub(r"^(and|or)\s+", "", t)
        t = re.sub(r"^(a|an|the)\s+", "", t)
        t = re.sub(r"^(small|medium|large|fresh|frozen|canned|pre\-made|plain|unsweetened|little|chopped|diced|sliced)\s+", "", t)
        t = re.sub(r"^(amount of|a small amount of|drizzle of|a drizzle of)\s+", "", t)
        t = re.sub(r"\s+", " ", t).strip(" ,.-")

        if not t or t in stop_exact:
            return ""

        # Canonical forms.
        replacements = {
            "whole wheat toast": "whole wheat bread",
            "plant based milk": "plant-based milk",
            "black salt": "black salt",
            "bell pepper": "bell peppers",
            "chickpea": "chickpeas",
            "lentil": "lentils",
            "bean": "beans",
            "plain greek yogurt": "greek yogurt",
            "rolled oats": "oats",
            "mixed berries": "berries",
        }
        t = replacements.get(t, t)

        # Drop pure flavor tokens.
        if t in {"vanilla", "chocolate", "unflavored"}:
            return ""

        return t

    def _normalize_grocery_item(raw_item):
        raw = str(raw_item or "").strip()
        if not raw:
            return []
        raw_l = raw.lower()
        if "leftover" in raw_l:
            return []

        parenthetical = re.findall(r"\(([^)]*)\)", raw)
        base = re.sub(r"\([^)]*\)", "", raw)
        base_l = base.strip().lower()

        # Only use parenthetical as main source for certain container phrases.
        if parenthetical and any(x in base_l for x in ["herbs", "spices", "mixed nuts", "mixed berries", "seasonal fruit"]):
            candidates = list(parenthetical)
        else:
            candidates = [base]

        results = []
        for cand in candidates:
            for part in re.split(r"[;,]", cand):
                part = part.strip()
                if not part:
                    continue

                options = [o.strip() for o in part.split(" or ")] if " or " in part.lower() else [part]
                clean_options = []
                for opt in options:
                    cleaned = _clean_token(opt)
                    if cleaned:
                        clean_options.append(cleaned)
                if not clean_options:
                    continue

                # Prefer longer concrete food names.
                clean_options.sort(key=lambda v: (v in {"water", "oil"}, len(v)))
                picked = clean_options[-1]
                if picked not in stop_exact:
                    results.append(picked)

        deduped = []
        seen = set()
        for r in results:
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        return deduped

    grocery = {}
    for day in plan_doc.get("days", []):
        for slot in day.get("slots", []):
            for item in slot.get("ingredients", []):
                for cleaned in _normalize_grocery_item(item):
                    grocery[cleaned] = grocery.get(cleaned, 0) + 1

    return [{"item": k, "count": v} for k, v in sorted(grocery.items(), key=lambda x: (-x[1], x[0]))]


def _name_key(value):
    return _normalize_token(value or "")


def _clean_doc_for_ai(doc):
    if not isinstance(doc, dict):
        return {}
    out = {}
    for k, v in doc.items():
        if k in {"_id", "user_id", "created_at", "updated_at"}:
            continue
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, ObjectId):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _recent_meal_context(user_id, limit=30):
    rows = []

    for m in db.meal_logs.find({"user_id": user_id}).sort("logged_at", -1).limit(limit):
        macros = m.get("macros") or {}
        rows.append(
            {
                "meal_name": str(m.get("meal_name") or "").strip(),
                "meal_type": str(m.get("meal_type") or "unspecified"),
                "source": str(m.get("source") or "manual"),
                "logged_at": m.get("logged_at").isoformat() if isinstance(m.get("logged_at"), datetime) else None,
                "calories_kcal": _safe_float(macros.get("calories_kcal")),
                "protein_g": _safe_float(macros.get("protein_g")),
                "carbs_g": _safe_float(macros.get("carbs_g")),
                "fat_g": _safe_float(macros.get("fat_g")),
                "notes": str(m.get("notes") or ""),
            }
        )

    if len(rows) < limit:
        rem = limit - len(rows)
        for m in db.collection.find({"user_id": user_id}).sort("created_at", -1).limit(rem):
            sj = m.get("analysis_json") or {}
            mid = sj.get("meal_identification")
            if isinstance(mid, dict):
                name = str(mid.get("name") or "").strip()
            else:
                name = str(mid or "").strip()
            tn = sj.get("total_nutrition") or {}
            rows.append(
                {
                    "meal_name": name,
                    "meal_type": str(m.get("meal_context") or "unspecified"),
                    "source": "legacy_image",
                    "logged_at": m.get("created_at").isoformat() if isinstance(m.get("created_at"), datetime) else None,
                    "calories_kcal": _safe_float(sj.get("calories_kcal", tn.get("calories", 0))),
                    "protein_g": _safe_float(sj.get("protein_g", tn.get("protein", 0))),
                    "carbs_g": _safe_float(sj.get("carbs_g", tn.get("carbs", 0))),
                    "fat_g": _safe_float(sj.get("fat_g", tn.get("fat", 0))),
                    "notes": "",
                }
            )

    return rows[:limit]


def _planner_ai_context(user_id):
    profile = db.user_profiles.find_one({"user_id": user_id}) or {}
    goals = db.nutrition_goals.find_one({"user_id": user_id}) or {}
    prefs = db.diet_preferences.find_one({"user_id": user_id}) or {}

    context: dict[str, object] = {
        "basic_info": _clean_doc_for_ai(profile),
        "goals": _clean_doc_for_ai(goals),
        "preferences": _clean_doc_for_ai(prefs),
    }
    context["recent_meals"] = _recent_meal_context(user_id, limit=30)
    return context


def _normalize_slots_map(slots):
    mapped = {}
    for s in slots or []:
        raw_slot = _normalize_token((s or {}).get("slot"))
        if raw_slot in {"breakfast", "morning"}:
            slot = "breakfast"
        elif raw_slot in {"lunch", "midday"}:
            slot = "lunch"
        elif raw_slot in {"dinner", "supper", "evening"}:
            slot = "dinner"
        else:
            continue
        mapped[slot] = {
            "slot": slot,
            "recipe_name": str((s or {}).get("recipe_name") or "Planned meal").strip(),
            "ingredients": [str(x).strip() for x in ((s or {}).get("ingredients") or []) if str(x).strip()],
            "notes": str((s or {}).get("notes") or "").strip(),
        }
    return mapped


def _planner_ingredients_from_pool(recipe_name, recipes, recipe_pool):
    """Match AI meal name to saved recipes so we can recover when the model omits ingredients."""
    nk = _name_key(recipe_name)
    if not nk:
        return None
    for coll in (recipes, recipe_pool):
        for r in coll or []:
            if _name_key(r.get("name")) != nk:
                continue
            out = [str(x).strip() for x in (r.get("ingredients") or []) if str(x).strip()]
            if out:
                return out[:16]
    return None


def _plan_has_excessive_repetition(days, max_occurrences=2):
    counts = {}
    prev_by_slot = {}
    consecutive = 0

    for day in days or []:
        for slot in (day or {}).get("slots", []):
            slot_name = _normalize_token((slot or {}).get("slot"))
            name_key = _name_key((slot or {}).get("recipe_name"))
            if not name_key:
                continue
            counts[name_key] = counts.get(name_key, 0) + 1
            if counts[name_key] > max_occurrences:
                return True
            if slot_name and prev_by_slot.get(slot_name) == name_key:
                consecutive += 1
                if consecutive >= 2:
                    return True
            if slot_name:
                prev_by_slot[slot_name] = name_key
    return False


def _build_recipe_cost_index(recipes):
    idx = {}
    for r in recipes or []:
        key = _name_key(r.get("name"))
        if not key:
            continue
        cost = _safe_float(r.get("cost_per_serving"), 0)
        if cost > 0:
            idx[key] = cost
    return idx


def _plan_violates_budget(days, cost_index, budget_per_meal):
    budget = _safe_float(budget_per_meal, 0)
    if budget <= 0 or not cost_index:
        return False

    for day in days or []:
        for slot in (day or {}).get("slots", []):
            name_key = _name_key((slot or {}).get("recipe_name"))
            if not name_key:
                continue
            cost = cost_index.get(name_key)
            # If budget is set and recipe has no known cost, reject plan for strictness.
            if cost is None:
                return True
            if cost > budget:
                return True
    return False


def _ai_generate_week_plan(week_start, ai_context, recipes):
    if not _v3_model:
        return None

    ai_context = ai_context or {}
    recent_meals_raw = ai_context.get("recent_meals")
    recent_meals = recent_meals_raw if isinstance(recent_meals_raw, list) else []
    prefs = ai_context.get("preferences") or {}
    budget_per_meal = _safe_float((prefs or {}).get("budget_per_meal"), 0)
    cost_index = _build_recipe_cost_index(recipes)

    recipe_pool = []
    for r in recipes[:30]:
        recipe_pool.append(
            {
                "id": str(r.get("_id")) if r.get("_id") else None,
                "name": r.get("name"),
                "description": r.get("description", ""),
                "ingredients": r.get("ingredients", [])[:8],
                "steps": r.get("steps", [])[:5],
                "servings": r.get("servings"),
                "diet_tags": r.get("diet_tags", [])[:4],
                "cook_time_min": r.get("cook_time_min"),
                "cost_per_serving": r.get("cost_per_serving"),
                "nutrition": r.get("nutrition", {}),
            }
        )

    target_dates = [(week_start + timedelta(days=i)).date().isoformat() for i in range(7)]
    prompt = f"""
You are generating a practical weekly meal plan.
Return ONLY valid JSON with this structure:
{{
  "days": [
    {{"date":"YYYY-MM-DD","slots":[
      {{"slot":"breakfast","recipe_name":"...","ingredients":["..."],"notes":"..."}},
      {{"slot":"lunch","recipe_name":"...","ingredients":["..."],"notes":"..."}},
      {{"slot":"dinner","recipe_name":"...","ingredients":["..."],"notes":"..."}}
    ]}}
  ]
}}

Rules:
- Exactly 7 days, with dates exactly matching: {json.dumps(target_dates)}
- Use all user context constraints (health, goals, allergies, restrictions, budget, prep time, timing, skill, living situation).
- Prefer recipe names from recipe_pool when possible.
- If recipe_pool is empty, generate meals from your nutrition knowledge using user context.
- Increase variety and avoid repeating the same meal all week.
- Do not use any recipe more than 2 times for the week.
- Do not repeat the same recipe in the same slot on consecutive days.
- Consider recent meals to avoid immediate repetition.
- Keep notes concise.
- If budget_per_meal is set (>0), each chosen meal should stay within that budget.
- If budget_per_meal is set and recipe_pool has costs, choose meals strictly from recipe_pool so costs are enforceable.
- Use recipe description and steps context from recipe_pool to improve plan quality.
- Every slot MUST include "ingredients" as a JSON array with at least 3 non-empty strings (required for grocery list).

User profile and goals context (from form):
{json.dumps(ai_context.get("basic_info") or {})}
{json.dumps(ai_context.get("goals") or {})}
{json.dumps(ai_context.get("preferences") or {})}

Recent meals:
{json.dumps(recent_meals)}

Recipe pool:
{json.dumps(recipe_pool)}
"""

    res, _ = _gemini_generate(prompt)
    raw_text = getattr(res, "text", "") if res else ""
    print(f"[PLANNER DEBUG] gemini response length: {len(raw_text)}, first 200 chars: {raw_text[:200]!r}")
    parsed = _parse_json_from_text(raw_text)
    if not parsed:
        print("[PLANNER DEBUG] failed: _parse_json_from_text returned empty")
        return None

    if not isinstance(parsed, dict):
        print(f"[PLANNER DEBUG] failed: parsed is not dict, got {type(parsed)}")
        return None
    ai_days = parsed.get("days")
    if not isinstance(ai_days, list):
        print(f"[PLANNER DEBUG] failed: 'days' is not a list, got {type(ai_days)}")
        return None

    if len(ai_days) != 7:
        print(f"[PLANNER DEBUG] failed: expected 7 days, got {len(ai_days)}")
        return None

    by_date = {}
    for day in ai_days:
        d = str((day or {}).get("date") or "").strip()
        if not d:
            continue
        by_date[d] = _normalize_slots_map((day or {}).get("slots") or [])

    out_days = []
    for date_key in target_dates:
        ai_slots = by_date.get(date_key, {})

        slots = []
        for slot_name in ["breakfast", "lunch", "dinner"]:
            slot = ai_slots.get(slot_name)
            if not slot:
                print(f"[PLANNER DEBUG] failed: missing slot '{slot_name}' for date {date_key}")
                return None
            ings = slot.get("ingredients") if isinstance(slot.get("ingredients"), list) else []
            if not ings:
                recovered = _planner_ingredients_from_pool(
                    slot.get("recipe_name"), recipes, recipe_pool
                )
                if recovered:
                    slot["ingredients"] = recovered
                    print(
                        f"[PLANNER DEBUG] repaired empty ingredients for '{slot_name}' on {date_key} from recipe pool"
                    )
                else:
                    slot["ingredients"] = [
                        "Protein of choice",
                        "Vegetables or salad",
                        "Starch or grain as needed",
                        "Seasonings and oil to taste",
                    ]
                    print(
                        f"[PLANNER DEBUG] repaired empty ingredients for '{slot_name}' on {date_key} with defaults"
                    )
            if not slot.get("notes"):
                slot["notes"] = "Personalized for your goals and preferences."
            slots.append(slot)

        if len(slots) != 3:
            return None
        out_days.append({"date": date_key, "slots": slots})

    if len(out_days) != 7:
        print(f"[PLANNER DEBUG] failed: out_days has {len(out_days)} entries")
        return None
    if _plan_has_excessive_repetition(out_days, max_occurrences=2):
        print("[PLANNER DEBUG] failed: excessive meal repetition")
        return None
    if _plan_violates_budget(out_days, cost_index, budget_per_meal):
        print("[PLANNER DEBUG] failed: plan violates budget")
        return None
    return out_days


def migrate_user_history_to_meal_logs(user_id):
    already = db.migration_state.find_one({"name": f"analysis_history_to_meal_logs:{str(user_id)}"})
    legacy_count = db.collection.count_documents({"user_id": user_id})
    linked_count = db.meal_logs.count_documents({"user_id": user_id, "legacy_analysis_id": {"$exists": True, "$ne": None}})
    if already and already.get("completed") and linked_count >= legacy_count:
        return {"migrated": 0, "skipped": 0, "already_completed": True}

    cursor = db.collection.find({"user_id": user_id})
    migrated = 0
    skipped = 0
    now = datetime.now(timezone.utc)

    for legacy in cursor:
        exists = db.meal_logs.find_one({"legacy_analysis_id": legacy.get("_id")})
        if exists:
            skipped += 1
            continue

        analysis_json = legacy.get("analysis_json") or {}
        macros = _extract_macros(analysis_json)

        logged_at = legacy.get("created_at")
        if not logged_at:
            try:
                logged_at = datetime.fromisoformat((legacy.get("timestamp") or "").replace("Z", "+00:00"))
            except Exception:
                logged_at = now

        meal = {
            "schema_version": 3,
            "user_id": user_id,
            "source": "image",
            "meal_name": _meal_name_from_structured(analysis_json, "Migrated Meal"),
            "notes": "Migrated from legacy analysis history.",
            "diet_type": legacy.get("dietary_goal") or "standard_american",
            "meal_type": legacy.get("meal_context") or "unspecified",
            "macros": macros,
            "image_base64": legacy.get("image_base64"),
            "barcode": None,
            "raw_input": None,
            "metadata": {
                "legacy_collection": "analysis_history",
                "legacy_analysis_id": str(legacy.get("_id")),
            },
            "legacy_analysis_id": legacy.get("_id"),
            "logged_at": logged_at,
            "created_at": now,
            "updated_at": now,
        }
        db.meal_logs.insert_one(meal)
        migrated += 1

    legacy_count_after = db.collection.count_documents({"user_id": user_id})
    linked_count_after = db.meal_logs.count_documents({"user_id": user_id, "legacy_analysis_id": {"$exists": True, "$ne": None}})

    db.migration_state.update_one(
        {"name": f"analysis_history_to_meal_logs:{str(user_id)}"},
        {
            "$set": {
                "completed": linked_count_after >= legacy_count_after,
                "migrated": migrated,
                "skipped": skipped,
                "legacy_count": legacy_count_after,
                "linked_count": linked_count_after,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    return {"migrated": migrated, "skipped": skipped, "already_completed": False}


def _get_user_context(user_id):
    prefs = db.diet_preferences.find_one({"user_id": user_id}) or {}
    goals = db.nutrition_goals.find_one({"user_id": user_id}) or {}
    profile = db.user_profiles.find_one({"user_id": user_id}) or {}
    return {
        "diet_type": prefs.get("diet_type") or "standard_american",
        "allergies": prefs.get("allergies", []),
        "food_restrictions": prefs.get("food_restrictions", []),
        "meal_timing_preference": prefs.get("meal_timing_preference"),
        "prep_time_limit": prefs.get("prep_time_limit"),
        "budget_per_meal": prefs.get("budget_per_meal"),
        "meal_prep_preference": prefs.get("meal_prep_preference"),
        "class_schedule": prefs.get("class_schedule", {}),
        "cooking_skill": prefs.get("cooking_skill"),
        "living_situation": prefs.get("living_situation"),
        "goal_type": goals.get("goal_type") or "maintain_weight",
        "target_weight_kg": goals.get("target_weight_kg"),
        "timeline_weeks": goals.get("timeline_weeks"),
        "secondary_goals": goals.get("secondary_goals", []),
        "daily_calories": goals.get("daily_calories") or 2000,
        "target_protein_g": goals.get("protein_grams"),
        "target_carbs_g": goals.get("carbs_grams"),
        "target_fat_g": goals.get("fat_grams"),
        "target_fiber_g": goals.get("fiber_grams"),
        "target_sodium_mg": goals.get("sodium_mg"),
        "target_sugar_g": goals.get("sugar_grams"),
        "height_cm": profile.get("height_cm"),
        "weight_kg": profile.get("weight_kg"),
        "age": profile.get("age"),
        "biological_sex": profile.get("biological_sex"),
        "activity_level": profile.get("activity_level"),
        "health_conditions": profile.get("health_conditions", []),
        "medications": profile.get("medications"),
        "supplements": profile.get("supplements", []),
    }


def _context_missing_fields(user_id):
    profile = db.user_profiles.find_one({"user_id": user_id}) or {}
    goals = db.nutrition_goals.find_one({"user_id": user_id}) or {}
    prefs = db.diet_preferences.find_one({"user_id": user_id}) or {}

    missing = []

    if not prefs.get("diet_type"):
        missing.append("diet_type")
    if not goals.get("goal_type"):
        missing.append("goal_type")
    if not profile.get("age"):
        missing.append("age")
    if not profile.get("height_cm"):
        missing.append("height_cm")
    if not profile.get("weight_kg"):
        missing.append("weight_kg")
    if not profile.get("biological_sex"):
        missing.append("biological_sex")
    if not profile.get("activity_level"):
        missing.append("activity_level")

    if prefs.get("prep_time_limit") in [None, "", 0]:
        missing.append("prep_time_limit")
    if prefs.get("budget_per_meal") in [None, "", 0]:
        missing.append("budget_per_meal")

    # Consider these complete if key exists, even if user intentionally left empty list.
    if "allergies" not in prefs:
        missing.append("allergies")
    if "food_restrictions" not in prefs:
        missing.append("food_restrictions")

    return missing


def _as_utc_aware(dt):
    """Mongo often stores naive UTC datetimes; compare everything as timezone-aware UTC."""
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _inventory_status(item, now=None):
    now = _as_utc_aware(now or datetime.now(timezone.utc))
    quantity = _safe_float(item.get("quantity"), 0)
    threshold = _safe_float(item.get("low_stock_threshold"), 1)
    expires_at = item.get("expires_at")
    if isinstance(expires_at, str):
        try:
            expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except Exception:
            expires_at = None
    days_left = None
    if expires_at and isinstance(expires_at, datetime):
        expires_at = _as_utc_aware(expires_at)
        delta = expires_at - now
        days_left = delta.days
        if days_left <= 3:
            return "expiring", days_left
    if quantity <= threshold:
        return "low_stock", days_left
    return "ok", days_left


@v3_bp.route("/inventory")
def inventory_page():
    return render_template("inventory.html")


@v3_bp.route("/planner")
def planner_page():
    return render_template("planner.html")


@v3_bp.route("/recipes")
def recipes_page():
    return render_template("recipes.html")


@v3_bp.route("/progress")
def progress_page():
    return render_template("progress.html")


@v3_bp.route("/coach")
def coach_page():
    if not _is_authed():
        return redirect(url_for("index", login=1, next=request.path))
    return render_template("coach.html")


@v3_bp.route("/social")
def social_page():
    if not _is_authed():
        return redirect(url_for("index", login=1, next=request.path))
    return render_template("social.html")


@v3_bp.route("/integrations")
def integrations_page():
    if not _is_authed():
        return redirect(url_for("index", login=1, next=request.path))
    return render_template("integrations.html")


@v3_bp.route("/api/v3/status")
def v3_status():
    return jsonify(
        {
            "success": True,
            "features": {
                "multi_logging": True,
                "planner": True,
                "recipes": True,
                "progress": True,
                "coach": True,
                "social": True,
                "integrations": True,
            },
            "ai_enabled": bool(_v3_model),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@v3_bp.route("/api/v3/context")
def v3_context():
    user_id, guest_sid, err = _actor_resolve()
    if err:
        return err

    if guest_sid:
        ctx = _guest_default_context()
        return jsonify(
            {
                "success": True,
                "guest_mode": True,
                "context": ctx,
                "missing": [],
                "completeness": 0,
                "guest_trial": guest_v3_trial_status(guest_sid),
            }
        )

    ctx = _get_user_context(user_id)
    required_keys = [
        "diet_type",
        "goal_type",
        "age",
        "height_cm",
        "allergies",
        "food_restrictions",
        "prep_time_limit",
        "budget_per_meal",
        "weight_kg",
        "biological_sex",
        "activity_level",
    ]
    missing = _context_missing_fields(user_id)
    completeness = max(0, min(100, int(((len(required_keys) - len(missing)) / len(required_keys)) * 100)))
    return jsonify(
        {
            "success": True,
            "guest_mode": False,
            "context": ctx,
            "missing": missing,
            "completeness": completeness,
        }
    )


@v3_bp.route("/api/v3/migrate", methods=["POST"])
def v3_migrate():
    user_id, err = _auth_guard()
    if err:
        return err
    stats = migrate_user_history_to_meal_logs(user_id)
    return jsonify({"success": True, "stats": stats})


@v3_bp.route("/api/v3/barcode/<barcode>")
def v3_barcode_lookup(barcode):
    user_id, guest_sid, err = _actor_resolve()
    if err:
        return err

    barcode = re.sub(r"\D", "", barcode or "")
    if len(barcode) < 8:
        return jsonify({"success": False, "error": "invalid_barcode", "message": "Enter at least 8 digits."}), 400

    cands = _barcode_candidates(barcode)
    cached = db.barcode_cache.find_one({"barcode": {"$in": cands}})
    if cached:
        doc = _serialize_oid(cached)
        doc.pop("raw", None)
        return jsonify({"success": True, "data": doc, "cached": True})

    fetched = _lookup_barcode_openfoodfacts(barcode)
    if not fetched:
        return jsonify(
            {
                "success": False,
                "error": "barcode_not_found",
                "message": "No product for this barcode in Open Food Facts. Check the digits or try Text / Manual.",
            }
        ), 404

    now = datetime.now(timezone.utc)
    fetched_doc = {
        **fetched,
        "created_at": now,
        "updated_at": now,
        "schema_version": 3,
        "created_by": user_id,
    }
    db.barcode_cache.update_one(
        {"barcode": barcode},
        {"$set": fetched_doc},
        upsert=True,
    )

    clean = dict(fetched_doc)
    clean.pop("raw", None)
    clean["created_by"] = str(user_id) if user_id else None
    return jsonify({"success": True, "data": clean, "cached": False})


@v3_bp.route("/api/v3/meals/log", methods=["POST"])
def v3_meals_log():
    user_id, guest_sid, err = _actor_resolve()
    if err:
        return err

    user_context = _get_user_context(user_id) if user_id else _guest_default_context()
    now = datetime.now(timezone.utc)

    source = request.form.get("source") if request.form else None
    if not source and request.is_json:
        source = (request.get_json(silent=True) or {}).get("source")
    source = (source or "manual").strip().lower()

    data = request.get_json(silent=True) if request.is_json else {}
    if not isinstance(data, dict):
        data = {}

    meal_type = (request.form.get("meal_type") if request.form else None) or data.get("meal_type") or "unspecified"
    notes = (request.form.get("notes") if request.form else None) or data.get("notes") or ""

    payload = {
        "source": source,
        "meal_type": meal_type,
        "diet_type": user_context.get("diet_type"),
        "notes": notes,
    }

    trial_reserved = False

    def _guest_trial_fail_response(msg):
        trial = guest_v3_trial_status(guest_sid) if guest_sid else None
        return (
            jsonify(
                {
                    "success": False,
                    "error": "guest_trial_exhausted",
                    "message": msg,
                    "guest_trial": trial,
                }
            ),
            403,
        )

    if guest_sid:
        if not try_reserve_guest_v3_trial(guest_sid):
            return _guest_trial_fail_response(
                f"You have used all {GUEST_V3_TRIAL_LIMIT} free meal logs. Sign in with Google to continue."
            )
        trial_reserved = True

    def _refund_trial():
        if trial_reserved and guest_sid:
            refund_guest_v3_trial(guest_sid)

    if source == "text":
        text_input = (request.form.get("text_input") if request.form else None) or data.get("text_input") or data.get("text")
        if not text_input:
            _refund_trial()
            return jsonify({"success": False, "error": "text_input_required"}), 400
        structured = _ai_structured_from_text(text_input, user_context)
        if not structured:
            _refund_trial()
            return jsonify({"success": False, "error": "ai_generation_failed", "message": "Gemini could not parse meal text."}), 502
        payload.update(
            {
                "meal_name": structured.get("meal_name") or text_input[:50],
                "calories_kcal": structured.get("calories_kcal"),
                "protein_g": structured.get("protein_g"),
                "carbs_g": structured.get("carbs_g"),
                "fat_g": structured.get("fat_g"),
                "fiber_g": structured.get("fiber_g"),
                "sodium_mg": structured.get("sodium_mg"),
                "raw_input": text_input,
                "metadata": {"structured": structured},
            }
        )

    elif source == "barcode":
        barcode = (request.form.get("barcode") if request.form else None) or data.get("barcode")
        if not barcode:
            _refund_trial()
            return jsonify(
                {"success": False, "error": "barcode_required", "message": "Enter a barcode number."}
            ), 400
        barcode = re.sub(r"\D", "", barcode)
        bc_candidates = _barcode_candidates(barcode)
        cached = db.barcode_cache.find_one({"barcode": {"$in": bc_candidates}})
        if not cached:
            fetched = _lookup_barcode_openfoodfacts(barcode)
            if not fetched:
                _refund_trial()
                return jsonify(
                    {
                        "success": False,
                        "error": "barcode_not_found",
                        "message": "No product for this barcode in Open Food Facts. Check the digits or try Text / Manual.",
                    }
                ), 404
            cached = {
                **fetched,
                "barcode": barcode,
                "created_at": now,
                "updated_at": now,
                "schema_version": 3,
                "created_by": user_id,
            }
            db.barcode_cache.update_one({"barcode": barcode}, {"$set": cached}, upsert=True)

        servings = _safe_float((request.form.get("servings") if request.form else None) or data.get("servings") or 1, 1)
        servings = max(0.1, servings)
        payload.update(
            {
                "meal_name": cached.get("name") or "Packaged food",
                "calories_kcal": _safe_float(cached.get("calories_kcal")) * servings,
                "protein_g": _safe_float(cached.get("protein_g")) * servings,
                "carbs_g": _safe_float(cached.get("carbs_g")) * servings,
                "fat_g": _safe_float(cached.get("fat_g")) * servings,
                "fiber_g": _safe_float(cached.get("fiber_g")) * servings,
                "sodium_mg": _safe_float(cached.get("sodium_mg")) * servings,
                "barcode": barcode,
                "raw_input": cached.get("name"),
                "metadata": {
                    "brand": cached.get("brand"),
                    "servings": servings,
                    "serving_size": cached.get("serving_size"),
                },
            }
        )

    elif source == "image":
        img = None
        image_url = (request.form.get("image_url") if request.form else None) or data.get("image_url")
        image_file = request.files.get("image_file") if request.files else None
        try:
            img = _image_from_request(image_file, image_url)
        except Exception as e:
            _refund_trial()
            return jsonify({"success": False, "error": "image_processing_failed", "message": str(e)}), 400

        if img is None:
            _refund_trial()
            return jsonify({"success": False, "error": "image_required"}), 400

        structured = _ai_structured_from_image(img, user_context)
        if not structured:
            _refund_trial()
            return jsonify({"success": False, "error": "ai_generation_failed", "message": "Gemini could not analyze meal image."}), 502
        payload.update(
            {
                "meal_name": structured.get("meal_name") or "Meal from image",
                "calories_kcal": structured.get("calories_kcal"),
                "protein_g": structured.get("protein_g"),
                "carbs_g": structured.get("carbs_g"),
                "fat_g": structured.get("fat_g"),
                "fiber_g": structured.get("fiber_g"),
                "sodium_mg": structured.get("sodium_mg"),
                "image_base64": _to_base64_jpeg(img),
                "raw_input": image_url or (image_file.filename if image_file else "image_upload"),
                "metadata": {"structured": structured},
            }
        )

    else:
        payload.update(
            {
                "meal_name": data.get("meal_name") or (request.form.get("meal_name") if request.form else None) or "Manual meal",
                "calories_kcal": data.get("calories_kcal") or (request.form.get("calories_kcal") if request.form else None),
                "protein_g": data.get("protein_g") or (request.form.get("protein_g") if request.form else None),
                "carbs_g": data.get("carbs_g") or (request.form.get("carbs_g") if request.form else None),
                "fat_g": data.get("fat_g") or (request.form.get("fat_g") if request.form else None),
                "fiber_g": data.get("fiber_g") or (request.form.get("fiber_g") if request.form else None),
                "sodium_mg": data.get("sodium_mg") or (request.form.get("sodium_mg") if request.form else None),
                "raw_input": data.get("raw_input") or "manual_entry",
            }
        )

    if request.form and request.form.get("logged_at"):
        try:
            payload["logged_at"] = datetime.fromisoformat(str(request.form.get("logged_at")).replace("Z", "+00:00"))
        except Exception:
            payload["logged_at"] = now
    elif data.get("logged_at"):
        try:
            payload["logged_at"] = datetime.fromisoformat(str(data.get("logged_at")).replace("Z", "+00:00"))
        except Exception:
            payload["logged_at"] = now

    try:
        macro_score = compute_macro_adherence_10pt(
            _safe_float(payload.get('calories_kcal')),
            _safe_float(payload.get('carbs_g')),
            _safe_float(payload.get('protein_g')),
            _safe_float(payload.get('fat_g')),
            user_context.get('diet_type') or 'standard_american',
        )
    except Exception:
        macro_score = {"score": None, "explanation": "computation_error"}
    payload['macro_adherence'] = macro_score

    try:
        if user_id:
            meal_doc = _save_meal_log(payload, user_id=user_id)
        else:
            meal_doc = _save_meal_log(payload, guest_session_id=guest_sid)
    except Exception:
        _refund_trial()
        raise

    meal_doc.pop("guest_session_id", None)
    out = {"success": True, "meal": meal_doc}
    if guest_sid:
        out["guest_trial"] = guest_v3_trial_status(guest_sid)
    return jsonify(out)


@v3_bp.route("/api/v3/meals/<meal_id>", methods=["PATCH", "DELETE"])
def v3_meals_patch_or_delete(meal_id):
    user_id, err = _auth_guard()
    if err:
        return err
    try:
        oid = ObjectId(meal_id)
    except Exception:
        return jsonify({"success": False, "error": "invalid_meal_id"}), 400

    doc = db.meal_logs.find_one({"_id": oid, "user_id": user_id})
    if not doc:
        return jsonify({"success": False, "error": "meal_not_found"}), 404

    if request.method == "DELETE":
        db.meal_logs.delete_one({"_id": oid, "user_id": user_id})
        return jsonify({"success": True})

    body = request.get_json(silent=True) or {}
    text_input = str(body.get("text_input") or "").strip()
    if not text_input:
        return jsonify({"success": False, "error": "text_input_required"}), 400

    meal_type = str(body.get("meal_type") or doc.get("meal_type") or "unspecified").strip() or "unspecified"
    user_context = _get_user_context(user_id)
    structured = _ai_structured_from_text(text_input, user_context)
    if not structured:
        return jsonify(
            {"success": False, "error": "ai_generation_failed", "message": "Gemini could not parse meal text."}
        ), 502

    try:
        macro_score = compute_macro_adherence_10pt(
            _safe_float(structured.get("calories_kcal")),
            _safe_float(structured.get("protein_g")),
            _safe_float(structured.get("carbs_g")),
            _safe_float(structured.get("fat_g")),
            user_context.get("diet_type") or "standard_american",
        )
    except Exception:
        macro_score = {"score": None, "explanation": "computation_error"}

    now = datetime.now(timezone.utc)
    logged_at = doc.get("logged_at") or now

    update_doc = {
        "source": "text",
        "meal_type": meal_type,
        "diet_type": user_context.get("diet_type") or doc.get("diet_type") or "standard_american",
        "meal_name": structured.get("meal_name") or text_input[:80],
        "notes": str(structured.get("notes") or doc.get("notes") or ""),
        "macros": {
            "calories_kcal": _safe_float(structured.get("calories_kcal")),
            "protein_g": _safe_float(structured.get("protein_g")),
            "carbs_g": _safe_float(structured.get("carbs_g")),
            "fat_g": _safe_float(structured.get("fat_g")),
            "fiber_g": _safe_float(structured.get("fiber_g")),
            "sodium_mg": _safe_float(structured.get("sodium_mg")),
        },
        "raw_input": text_input,
        "metadata": {"structured": structured},
        "barcode": None,
        "image_base64": None,
        "personalization": {"macro_adherence": macro_score},
        "updated_at": now,
        "logged_at": logged_at,
    }
    db.meal_logs.update_one({"_id": oid, "user_id": user_id}, {"$set": update_doc})
    updated = db.meal_logs.find_one({"_id": oid})
    return jsonify({"success": True, "meal": _serialize_oid(updated)})


@v3_bp.route("/api/v3/meals")
def v3_meals_list():
    user_id, err = _auth_guard()
    if err:
        return err

    days = _safe_int(request.args.get("days", 14), 14)
    days = max(1, min(days, 180))
    start = datetime.now(timezone.utc) - timedelta(days=days)

    start_raw = request.args.get("start")
    if start_raw:
        try:
            start = datetime.fromisoformat(str(start_raw).replace("Z", "+00:00"))
        except Exception:
            pass

    query = {"user_id": user_id, "logged_at": {"$gte": start}}
    meals = list(db.meal_logs.find(query).sort("logged_at", -1).limit(400))
    meals = [_serialize_oid(m) for m in meals]
    for m in meals:
        if isinstance(m.get("logged_at"), datetime):
            m["logged_at"] = m["logged_at"].isoformat()
        if isinstance(m.get("created_at"), datetime):
            m["created_at"] = m["created_at"].isoformat()
        if isinstance(m.get("updated_at"), datetime):
            m["updated_at"] = m["updated_at"].isoformat()

    return jsonify({"success": True, "count": len(meals), "meals": meals})


@v3_bp.route("/api/v3/meals/by-date")
def v3_meals_by_date():
    user_id, guest_sid, err = _actor_resolve()
    if err:
        return err

    tz_name = request.args.get("tz") or "UTC"
    day_str = request.args.get("date")

    try:
        start_utc, end_utc, local_day = _day_bounds_in_utc(day_str, tz_name)
    except Exception:
        tz_name = "UTC"
        start_utc, end_utc, local_day = _day_bounds_in_utc(day_str, tz_name)

    if user_id:
        query = {
            "user_id": user_id,
            "logged_at": {
                "$gte": start_utc,
                "$lt": end_utc,
            },
        }
    else:
        query = {
            "guest_session_id": guest_sid,
            "logged_at": {
                "$gte": start_utc,
                "$lt": end_utc,
            },
        }

    cursor = db.meal_logs.find(query).sort("logged_at", 1)

    meals = []
    totals = {
        "calories_kcal": 0.0,
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fat_g": 0.0,
    }

    tz = ZoneInfo(tz_name)
    for m in cursor:
        macros = m.get("macros") or {}
        totals["calories_kcal"] += _safe_float(macros.get("calories_kcal"))
        totals["protein_g"] += _safe_float(macros.get("protein_g"))
        totals["carbs_g"] += _safe_float(macros.get("carbs_g"))
        totals["fat_g"] += _safe_float(macros.get("fat_g"))

        logged_at = m.get("logged_at")
        local_iso = None
        local_time = None
        if isinstance(logged_at, datetime):
            local_dt = logged_at.astimezone(tz)
            local_iso = local_dt.isoformat()
            local_time = local_dt.strftime("%I:%M %p").lstrip("0")

        meals.append(
            {
                "id": str(m.get("_id")),
                "meal_name": m.get("meal_name") or "Meal",
                "source": m.get("source") or "manual",
                "meal_type": m.get("meal_type") or "unspecified",
                "logged_at": logged_at.isoformat() if isinstance(logged_at, datetime) else None,
                "local_logged_at": local_iso,
                "local_time": local_time,
                "macros": {
                    "calories_kcal": _safe_float(macros.get("calories_kcal")),
                    "protein_g": _safe_float(macros.get("protein_g")),
                    "carbs_g": _safe_float(macros.get("carbs_g")),
                    "fat_g": _safe_float(macros.get("fat_g")),
                },
                "notes": m.get("notes", ""),
            }
        )

    payload = {
        "success": True,
        "date": local_day.isoformat(),
        "timezone": tz_name,
        "count": len(meals),
        "totals": {k: round(v, 2) for k, v in totals.items()},
        "meals": meals,
    }
    if guest_sid:
        payload["guest_mode"] = True
        payload["guest_trial"] = guest_v3_trial_status(guest_sid)
    return jsonify(payload)


@v3_bp.route("/api/v3/recipes", methods=["GET", "POST"])
def v3_recipes():
    user_id, err = _auth_guard()
    if err:
        return err

    if request.method == "GET":
        include_public = request.args.get("include_public", "1") == "1"
        diet_tag = request.args.get("diet_tag")
        user_context = _get_user_context(user_id)
        preferred_diet = _normalize_token(user_context.get("diet_type"))

        visibility = []
        visibility.append({"user_id": user_id})
        if include_public:
            visibility.append({"is_public": True})

        query = {"$or": visibility}
        docs = list(db.recipes.find(query).sort("created_at", -1).limit(600))

        filtered = []
        for d in docs:
            own_recipe = d.get("user_id") == user_id
            tags = [_normalize_token(t) for t in (d.get("diet_tags") or [])]
            if diet_tag and _normalize_token(diet_tag) not in tags:
                continue

            # Always show user's own recipes (even if they don't match current profile filters).
            if own_recipe:
                filtered.append(d)
                continue

            # For public recipes, apply profile-aware matching.
            if _recipe_matches_context(d, user_context):
                filtered.append(d)

        def _sort_key(doc):
            tags = [_normalize_token(t) for t in (doc.get("diet_tags") or [])]
            own = 1 if doc.get("user_id") == user_id else 0
            diet_match = 1 if preferred_diet and preferred_diet in tags else 0
            return (own, diet_match, doc.get("created_at") or datetime.min)

        docs = sorted(filtered, key=_sort_key, reverse=True)[:300]

        items = []
        for d in docs:
            s = _serialize_oid(d)
            if isinstance(s.get("created_at"), datetime):
                s["created_at"] = s["created_at"].isoformat()
            if isinstance(s.get("updated_at"), datetime):
                s["updated_at"] = s["updated_at"].isoformat()
            items.append(s)
        return jsonify({"success": True, "recipes": items})

    body = request.get_json(silent=True) or {}
    now = datetime.now(timezone.utc)
    recipe = {
        "schema_version": 3,
        "user_id": user_id,
        "name": (body.get("name") or "Untitled Recipe").strip(),
        "description": body.get("description") or "",
        "diet_tags": [_normalize_token(x) for x in (body.get("diet_tags") or [])],
        "ingredients": body.get("ingredients") or [],
        "steps": body.get("steps") or [],
        "servings": _safe_int(body.get("servings", 1), 1),
        "cook_time_min": _safe_int(body.get("cook_time_min", 20), 20),
        "cost_per_serving": _safe_float(body.get("cost_per_serving", 0), 0),
        "nutrition": {
            "calories_kcal": _safe_float((body.get("nutrition") or {}).get("calories_kcal", body.get("calories_kcal", 0))),
            "protein_g": _safe_float((body.get("nutrition") or {}).get("protein_g", body.get("protein_g", 0))),
            "carbs_g": _safe_float((body.get("nutrition") or {}).get("carbs_g", body.get("carbs_g", 0))),
            "fat_g": _safe_float((body.get("nutrition") or {}).get("fat_g", body.get("fat_g", 0))),
        },
        "is_public": bool(body.get("is_public", False)),
        "created_at": now,
        "updated_at": now,
    }
    try:
        result = db.recipes.insert_one(recipe)
        recipe["_id"] = result.inserted_id
        return jsonify({"success": True, "recipe": _serialize_oid(recipe)})
    except DuplicateKeyError:
        return jsonify(
            {
                "success": False,
                "error": "recipe_name_exists",
                "message": "A recipe with this name already exists for your account. Use a different name.",
            }
        ), 409


@v3_bp.route("/api/v3/recipes/bulk-delete", methods=["POST"])
def v3_recipes_bulk_delete():
    user_id, err = _auth_guard()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    raw_ids = body.get("ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        return jsonify({"success": False, "error": "ids_required"}), 400
    oids = []
    for rid in raw_ids[:500]:
        try:
            oids.append(ObjectId(str(rid)))
        except Exception:
            continue
    if not oids:
        return jsonify({"success": False, "error": "no_valid_ids"}), 400
    result = db.recipes.delete_many({"user_id": user_id, "_id": {"$in": oids}})
    return jsonify({"success": True, "deleted": result.deleted_count})


@v3_bp.route("/api/v3/recipes/<recipe_id>", methods=["PUT", "DELETE"])
def v3_recipe_detail(recipe_id):
    user_id, err = _auth_guard()
    if err:
        return err
    try:
        oid = ObjectId(recipe_id)
    except Exception:
        return jsonify({"success": False, "error": "invalid_recipe_id"}), 400

    existing = db.recipes.find_one({"_id": oid})
    if not existing:
        return jsonify({"success": False, "error": "not_found"}), 404
    if existing.get("user_id") != user_id:
        return jsonify({"success": False, "error": "forbidden"}), 403

    if request.method == "DELETE":
        db.recipes.delete_one({"_id": oid})
        return jsonify({"success": True})

    body = request.get_json(silent=True) or {}
    updates = {
        "name": (body.get("name") or existing.get("name") or "Untitled Recipe").strip(),
        "description": body.get("description", existing.get("description", "")),
        "diet_tags": [_normalize_token(x) for x in body.get("diet_tags", existing.get("diet_tags", []))],
        "ingredients": body.get("ingredients", existing.get("ingredients", [])),
        "steps": body.get("steps", existing.get("steps", [])),
        "servings": _safe_int(body.get("servings", existing.get("servings", 1)), 1),
        "cook_time_min": _safe_int(body.get("cook_time_min", existing.get("cook_time_min", 20)), 20),
        "cost_per_serving": _safe_float(body.get("cost_per_serving", existing.get("cost_per_serving", 0)), 0),
        "nutrition": body.get("nutrition", existing.get("nutrition", {})),
        "is_public": bool(body.get("is_public", existing.get("is_public", False))),
        "updated_at": datetime.now(timezone.utc),
    }
    try:
        db.recipes.update_one({"_id": oid}, {"$set": updates})
    except DuplicateKeyError:
        return jsonify(
            {
                "success": False,
                "error": "recipe_name_exists",
                "message": "A recipe with this name already exists for your account. Use a different name.",
            }
        ), 409

    latest = db.recipes.find_one({"_id": oid})
    return jsonify({"success": True, "recipe": _serialize_oid(latest)})


@v3_bp.route("/api/v3/inventory", methods=["GET", "POST"])
def v3_inventory():
    user_id, err = _auth_guard()
    if err:
        return err

    if request.method == "GET":
        location = (request.args.get("location") or "").strip().lower()
        query = {"user_id": user_id}
        if location in {"pantry", "fridge", "freezer"}:
            query["location"] = location
        docs = list(db.inventory_items.find(query).sort("updated_at", -1).limit(1000))
        now = datetime.now(timezone.utc)
        out = []
        counts = {"low_stock": 0, "expiring": 0, "ok": 0}
        for d in docs:
            status, days_left = _inventory_status(d, now=now)
            counts[status] = counts.get(status, 0) + 1
            row = _serialize_oid(d)
            if isinstance(row.get("expires_at"), datetime):
                row["expires_at"] = row["expires_at"].isoformat()
            if isinstance(row.get("created_at"), datetime):
                row["created_at"] = row["created_at"].isoformat()
            if isinstance(row.get("updated_at"), datetime):
                row["updated_at"] = row["updated_at"].isoformat()
            row["status"] = status
            row["days_to_expiry"] = days_left
            out.append(row)
        return jsonify({"success": True, "items": out, "summary": {"count": len(out), **counts}})

    # POST
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"success": False, "error": "name_required"}), 400
    location = (body.get("location") or "pantry").strip().lower()
    if location not in {"pantry", "fridge", "freezer"}:
        location = "pantry"
    now = datetime.now(timezone.utc)
    expires_at = None
    exp_val = body.get("expires_at")
    if exp_val:
        try:
            expires_at = datetime.fromisoformat(str(exp_val).replace("Z", "+00:00"))
        except Exception:
            return jsonify({"success": False, "error": "invalid_expiry"}), 400

    item = {
        "user_id": user_id,
        "name": name,
        "location": location,
        "category": (body.get("category") or "").strip(),
        "quantity": _safe_float(body.get("quantity"), 1),
        "unit": (body.get("unit") or "pcs").strip(),
        "low_stock_threshold": _safe_float(body.get("low_stock_threshold"), 1),
        "expires_at": expires_at,
        "notes": (body.get("notes") or "").strip(),
        "created_at": now,
        "updated_at": now,
    }
    ins = db.inventory_items.insert_one(item)
    item["_id"] = ins.inserted_id
    row = _serialize_oid(item)
    if isinstance(row.get("expires_at"), datetime):
        row["expires_at"] = row["expires_at"].isoformat()
    row["status"], row["days_to_expiry"] = _inventory_status(item, now=now)
    return jsonify({"success": True, "item": row})


@v3_bp.route("/api/v3/inventory/bulk-delete", methods=["POST"])
def v3_inventory_bulk_delete():
    user_id, err = _auth_guard()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    raw_ids = body.get("ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        return jsonify({"success": False, "error": "ids_required"}), 400
    oids = []
    for rid in raw_ids[:500]:
        try:
            oids.append(ObjectId(str(rid)))
        except Exception:
            continue
    if not oids:
        return jsonify({"success": False, "error": "no_valid_ids"}), 400
    result = db.inventory_items.delete_many({"user_id": user_id, "_id": {"$in": oids}})
    return jsonify({"success": True, "deleted": result.deleted_count})


@v3_bp.route("/api/v3/inventory/<item_id>", methods=["PUT", "DELETE"])
def v3_inventory_item(item_id):
    user_id, err = _auth_guard()
    if err:
        return err
    try:
        oid = ObjectId(item_id)
    except Exception:
        return jsonify({"success": False, "error": "invalid_item_id"}), 400

    existing = db.inventory_items.find_one({"_id": oid, "user_id": user_id})
    if not existing:
        return jsonify({"success": False, "error": "not_found"}), 404

    if request.method == "DELETE":
        db.inventory_items.delete_one({"_id": oid, "user_id": user_id})
        return jsonify({"success": True})

    body = request.get_json(silent=True) or {}
    updates = {}
    for key in ["name", "category", "unit", "notes"]:
        if key in body:
            val = (body.get(key) or "").strip()
            if key == "name" and not val:
                return jsonify({"success": False, "error": "name_required"}), 400
            updates[key] = val

    if "location" in body:
        loc = (body.get("location") or "").strip().lower()
        if loc in {"pantry", "fridge", "freezer"}:
            updates["location"] = loc

    if "quantity" in body:
        updates["quantity"] = _safe_float(body.get("quantity"), 1)
    if "low_stock_threshold" in body:
        updates["low_stock_threshold"] = _safe_float(body.get("low_stock_threshold"), 1)

    if "expires_at" in body:
        exp = body.get("expires_at")
        if exp is None:
            updates["expires_at"] = None
        else:
            try:
                updates["expires_at"] = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
            except Exception:
                return jsonify({"success": False, "error": "invalid_expiry"}), 400

    updates["updated_at"] = datetime.now(timezone.utc)
    db.inventory_items.update_one({"_id": oid, "user_id": user_id}, {"$set": updates})
    latest = db.inventory_items.find_one({"_id": oid, "user_id": user_id})
    row = _serialize_oid(latest)
    if isinstance(row.get("expires_at"), datetime):
        row["expires_at"] = row["expires_at"].isoformat()
    if isinstance(row.get("updated_at"), datetime):
        row["updated_at"] = row["updated_at"].isoformat()
    row["status"], row["days_to_expiry"] = _inventory_status(latest)
    return jsonify({"success": True, "item": row})


def _sanitize_inventory_saved_meals(raw):
    """Normalize AI meal suggestion payloads for storage (bounded size)."""
    if not isinstance(raw, list):
        return []
    out = []
    for m in raw[:25]:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()[:200]
        if not name:
            continue
        uses = m.get("uses")
        if not isinstance(uses, list):
            uses = []
        uses = [str(u).strip()[:200] for u in uses[:40] if str(u).strip()]
        out.append(
            {
                "name": name,
                "why": str(m.get("why") or "")[:4000],
                "uses": uses,
                "nutrition_note": str(m.get("nutrition_note") or "")[:4000],
            }
        )
    return out


@v3_bp.route("/api/v3/inventory/suggestions/saved", methods=["GET"])
def v3_inventory_suggestions_saved():
    user_id, err = _auth_guard()
    if err:
        return err
    coll = db.inventory_meal_suggestions
    doc = coll.find_one({"user_id": user_id})
    if not doc:
        return jsonify({"success": True, "meals": [], "updated_at": None})
    meals = doc.get("meals") or []
    if not isinstance(meals, list):
        meals = []
    updated = doc.get("updated_at")
    updated_iso = updated.isoformat() if isinstance(updated, datetime) else None
    return jsonify({"success": True, "meals": meals, "updated_at": updated_iso})


@v3_bp.route("/api/v3/inventory/suggestions", methods=["POST"])
def v3_inventory_suggestions():
    user_id, err = _auth_guard()
    if err:
        return err
    if not _v3_model:
        return jsonify({"success": False, "error": "ai_unavailable"}), 503

    ctx = _get_user_context(user_id)
    docs = list(db.inventory_items.find({"user_id": user_id}).limit(300))
    if not docs:
        return jsonify({"success": False, "error": "inventory_empty", "message": "Add inventory items first."}), 400

    now = datetime.now(timezone.utc)
    inv = []
    for d in docs:
        status, days_left = _inventory_status(d, now=now)
        inv.append(
            {
                "name": d.get("name"),
                "location": d.get("location"),
                "quantity": _safe_float(d.get("quantity"), 0),
                "unit": d.get("unit", "pcs"),
                "status": status,
                "days_to_expiry": days_left,
            }
        )

    prompt = (
        "Suggest exactly 5 practical meals for a student using this inventory. "
        "Prioritize expiring/low-stock balancing and minimal waste. "
        "Return strict JSON: {\"meals\":[{\"name\":\"...\",\"why\":\"...\",\"uses\":[\"...\"],\"nutrition_note\":\"...\"}]}\n"
        f"User context: {json.dumps(ctx)}\nInventory: {json.dumps(inv)}"
    )
    resp, gem_err = _gemini_generate(prompt)
    raw_text = (getattr(resp, "text", "") or "").strip() if resp else ""
    if not raw_text:
        if gem_err == "rate_limit":
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "ai_rate_limited",
                        "message": (
                            "Gemini free-tier or per-minute request limit was reached. "
                            "Wait about one minute and try again, or check quota and billing at "
                            "https://ai.google.dev/gemini-api/docs/rate-limits"
                        ),
                    }
                ),
                429,
            )
        return jsonify({"success": False, "error": "ai_generation_failed"}), 502
    data = _parse_json_from_text(raw_text)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "ai_parse_error"}), 502
    meals = data.get("meals", [])
    if not isinstance(meals, list):
        meals = []
    cleaned = _sanitize_inventory_saved_meals(meals)
    saved = False
    if cleaned:
        now = datetime.now(timezone.utc)
        db.inventory_meal_suggestions.update_one(
            {"user_id": user_id},
            {"$set": {"user_id": user_id, "meals": cleaned, "updated_at": now}, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        saved = True
    return jsonify({"success": True, "meals": cleaned, "saved": saved})


@v3_bp.route("/api/v3/planner/week", methods=["GET", "POST"])
def v3_planner_week():
    user_id, err = _auth_guard()
    if err:
        return err

    def _parse_week_start(raw):
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        except Exception:
            return None

    week_start_raw = request.args.get("week_start")
    if request.method == "POST":
        body_preview = request.get_json(silent=True) or {}
        week_start_raw = week_start_raw or body_preview.get("week_start")

    week_start = _parse_week_start(week_start_raw) or _week_start(datetime.now(timezone.utc))
    week_start = datetime(week_start.year, week_start.month, week_start.day, tzinfo=timezone.utc)

    if request.method == "GET":
        plan = db.meal_plans.find_one({"user_id": user_id, "week_start": week_start})
        grocery = db.grocery_lists.find_one({"user_id": user_id, "week_start": week_start})

        # Always normalize grocery from current plan so old noisy lists self-heal.
        if plan:
            normalized_items = _compute_grocery_from_plan(plan)
            now = datetime.now(timezone.utc)
            needs_update = (not grocery) or (grocery.get("items") != normalized_items)
            if needs_update:
                db.grocery_lists.update_one(
                    {"user_id": user_id, "week_start": week_start},
                    {
                        "$set": {
                            "schema_version": 3,
                            "user_id": user_id,
                            "week_start": week_start,
                            "items": normalized_items,
                            "updated_at": now,
                        },
                        "$setOnInsert": {"created_at": now},
                    },
                    upsert=True,
                )
                grocery = db.grocery_lists.find_one({"user_id": user_id, "week_start": week_start})

        if plan:
            plan = _serialize_oid(plan)
            plan["week_start"] = plan["week_start"].isoformat() if isinstance(plan.get("week_start"), datetime) else plan.get("week_start")
            plan["created_at"] = plan["created_at"].isoformat() if isinstance(plan.get("created_at"), datetime) else plan.get("created_at")
            plan["updated_at"] = plan["updated_at"].isoformat() if isinstance(plan.get("updated_at"), datetime) else plan.get("updated_at")
        if grocery:
            grocery = _serialize_oid(grocery)
            grocery["week_start"] = grocery["week_start"].isoformat() if isinstance(grocery.get("week_start"), datetime) else grocery.get("week_start")
            grocery["created_at"] = grocery["created_at"].isoformat() if isinstance(grocery.get("created_at"), datetime) else grocery.get("created_at")
            grocery["updated_at"] = grocery["updated_at"].isoformat() if isinstance(grocery.get("updated_at"), datetime) else grocery.get("updated_at")
        return jsonify({"success": True, "plan": plan, "grocery": grocery})

    body = request.get_json(silent=True) or {}
    days = body.get("days") or []
    now = datetime.now(timezone.utc)
    plan_doc = {
        "schema_version": 3,
        "user_id": user_id,
        "week_start": week_start,
        "days": days,
        "notes": body.get("notes") or "",
        "updated_at": now,
    }
    db.meal_plans.update_one(
        {"user_id": user_id, "week_start": week_start},
        {"$set": plan_doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    grocery_items = _compute_grocery_from_plan(plan_doc)
    db.grocery_lists.update_one(
        {"user_id": user_id, "week_start": week_start},
        {
            "$set": {
                "schema_version": 3,
                "user_id": user_id,
                "week_start": week_start,
                "items": grocery_items,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    return jsonify({"success": True, "grocery_items": grocery_items})


@v3_bp.route("/api/v3/planner/generate", methods=["POST"])
def v3_planner_generate():
    user_id, err = _auth_guard()
    if err:
        return err

    planner_required_fields = [
        "diet_type",
        "goal_type",
        "age",
        "height_cm",
        "weight_kg",
        "biological_sex",
        "activity_level",
    ]
    missing = _context_missing_fields(user_id)
    planner_missing = [f for f in missing if f in planner_required_fields]
    if planner_missing:
        return jsonify(
            {
                "success": False,
                "error": "profile_incomplete",
                "message": "Complete your profile setup before planner generation.",
                "missing": planner_missing,
                "setup_url": url_for("profile.setup"),
            }
        ), 400

    if not _v3_model:
        return jsonify(
            {
                "success": False,
                "error": "ai_unavailable",
                "message": "Planner generation requires AI and is currently unavailable.",
            }
        ), 503

    body = request.get_json(silent=True) or {}
    start = body.get("week_start")
    try:
        week_start = datetime.fromisoformat(str(start).replace("Z", "+00:00")) if start else _week_start(datetime.now(timezone.utc))
    except Exception:
        week_start = _week_start(datetime.now(timezone.utc))
    week_start = datetime(week_start.year, week_start.month, week_start.day, tzinfo=timezone.utc)

    user_context = _get_user_context(user_id)
    diet_type = user_context.get("diet_type") or "standard_american"
    ai_context = _planner_ai_context(user_id)
    pref_ctx_raw = ai_context.get("preferences")
    pref_ctx = pref_ctx_raw if isinstance(pref_ctx_raw, dict) else {}

    recipes = list(db.recipes.find({"$or": [{"user_id": user_id}, {"is_public": True}]}).limit(250))
    recipes = [r for r in recipes if _recipe_matches_context(r, user_context)]
    budget_per_meal = _safe_float(pref_ctx.get("budget_per_meal"), 0)
    enforce_budget_with_cost = budget_per_meal > 0 and any(_safe_float(r.get("cost_per_serving"), 0) > 0 for r in recipes)

    recent_meals_raw = ai_context.get("recent_meals")
    recent_meals = recent_meals_raw if isinstance(recent_meals_raw, list) else []
    recent_meals_count = len(recent_meals)
    ai_days = _ai_generate_week_plan(week_start, ai_context, recipes)
    if not ai_days:
        time.sleep(1.0 + random.random() * 0.5)
        ai_days = _ai_generate_week_plan(week_start, ai_context, recipes)
    if not ai_days:
        return jsonify(
            {
                "success": False,
                "error": "ai_generation_failed",
                "message": "AI could not generate a valid plan. Please try again in a moment.",
            }
        ), 502

    days = ai_days
    generation_source = "ai"

    now = datetime.now(timezone.utc)
    plan_doc = {
        "schema_version": 3,
        "user_id": user_id,
        "week_start": week_start,
        "diet_type": diet_type,
        "days": days,
        "generated": True,
        "generation_source": generation_source,
        "context_used": {
            "basic_info": ai_context.get("basic_info", {}),
            "goals": ai_context.get("goals", {}),
            "preferences": ai_context.get("preferences", {}),
            "recent_meals_count": recent_meals_count,
            "recipe_pool_size": len(recipes),
            "budget_per_meal": budget_per_meal,
            "budget_cost_enforced": enforce_budget_with_cost,
        },
        "updated_at": now,
    }
    db.meal_plans.update_one(
        {"user_id": user_id, "week_start": week_start},
        {"$set": plan_doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    grocery_items = _compute_grocery_from_plan(plan_doc)
    db.grocery_lists.update_one(
        {"user_id": user_id, "week_start": week_start},
        {
            "$set": {
                "schema_version": 3,
                "user_id": user_id,
                "week_start": week_start,
                "items": grocery_items,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )

    return jsonify(
        {
            "success": True,
            "week_start": week_start.isoformat(),
            "diet_type": diet_type,
            "generation_source": generation_source,
            "recent_meals_considered": recent_meals,
            "recipe_pool_size": len(recipes),
            "budget_per_meal": budget_per_meal,
            "budget_cost_enforced": enforce_budget_with_cost,
            "days": days,
            "grocery_items": grocery_items,
        }
    )


@v3_bp.route("/api/v3/progress/weight", methods=["GET", "POST"])
def v3_progress_weight():
    user_id, err = _auth_guard()
    if err:
        return err

    if request.method == "GET":
        docs = list(db.weight_logs.find({"user_id": user_id}).sort("date", 1).limit(365))
        rows = []
        for d in docs:
            rows.append(
                {
                    "id": str(d.get("_id")),
                    "date": d.get("date"),
                    "weight_kg": _safe_float(d.get("weight_kg")),
                    "weight_lbs": round(_safe_float(d.get("weight_kg")) * 2.20462, 1),
                    "notes": d.get("notes", ""),
                }
            )
        return jsonify({"success": True, "weights": rows})

    body = request.get_json(silent=True) or {}
    date = body.get("date") or datetime.now(timezone.utc).date().isoformat()
    weight_kg = _safe_float(body.get("weight_kg"), 0)
    if weight_kg <= 0:
        return jsonify({"success": False, "error": "invalid_weight"}), 400

    now = datetime.now(timezone.utc)
    db.weight_logs.update_one(
        {"user_id": user_id, "date": date},
        {
            "$set": {
                "schema_version": 3,
                "user_id": user_id,
                "date": date,
                "weight_kg": weight_kg,
                "notes": body.get("notes", ""),
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    return jsonify({"success": True})


@v3_bp.route("/api/v3/progress/summary")
def v3_progress_summary():
    user_id, err = _auth_guard()
    if err:
        return err

    days = _safe_int(request.args.get("days", 30), 30)
    days = max(7, min(days, 180))
    start = datetime.now(timezone.utc) - timedelta(days=days)

    meals = list(db.meal_logs.find({"user_id": user_id, "logged_at": {"$gte": start}}).sort("logged_at", 1))
    by_day = {}
    for m in meals:
        dt = m.get("logged_at") or m.get("created_at")
        if not isinstance(dt, datetime):
            continue
        key = dt.date().isoformat()
        row = by_day.setdefault(
            key,
            {
                "date": key,
                "calories_kcal": 0.0,
                "protein_g": 0.0,
                "carbs_g": 0.0,
                "fat_g": 0.0,
                "count": 0,
            },
        )
        macros = m.get("macros") or {}
        row["calories_kcal"] += _safe_float(macros.get("calories_kcal"))
        row["protein_g"] += _safe_float(macros.get("protein_g"))
        row["carbs_g"] += _safe_float(macros.get("carbs_g"))
        row["fat_g"] += _safe_float(macros.get("fat_g"))
        row["count"] += 1

    points = [by_day[k] for k in sorted(by_day.keys())]
    total_days_logged = len(points)
    total_meals = sum(p["count"] for p in points)
    avg_calories = round(sum(p["calories_kcal"] for p in points) / total_days_logged, 1) if total_days_logged else 0
    avg_protein = round(sum(p["protein_g"] for p in points) / total_days_logged, 1) if total_days_logged else 0

    weights = list(db.weight_logs.find({"user_id": user_id}).sort("date", 1).limit(365))
    weight_series = [{"date": w.get("date"), "weight_kg": _safe_float(w.get("weight_kg"))} for w in weights]

    start_weight = weight_series[0]["weight_kg"] if weight_series else None
    end_weight = weight_series[-1]["weight_kg"] if weight_series else None
    delta_weight = round(end_weight - start_weight, 2) if start_weight is not None and end_weight is not None else None

    return jsonify(
        {
            "success": True,
            "window_days": days,
            "kpis": {
                "days_logged": total_days_logged,
                "meals_logged": total_meals,
                "avg_daily_calories": avg_calories,
                "avg_daily_protein_g": avg_protein,
                "weight_delta_kg": delta_weight,
            },
            "daily": points,
            "weight_series": weight_series,
        }
    )


def _coach_history_doc(user_id):
    doc = db.chat_sessions.find_one({"user_id": user_id, "kind": "coach"})
    if doc:
        return doc
    now = datetime.now(timezone.utc)
    base = {
        "schema_version": 3,
        "user_id": user_id,
        "kind": "coach",
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    inserted = db.chat_sessions.insert_one(base)
    base["_id"] = inserted.inserted_id
    return base


def _coach_context_payload(user_id):
    user_doc = db.users.find_one({"_id": user_id}) or {}
    ai_context = _planner_ai_context(user_id)
    basic_info_raw = ai_context.get("basic_info")
    goals_raw = ai_context.get("goals")
    preferences_raw = ai_context.get("preferences")
    recent_meals_raw = ai_context.get("recent_meals")

    basic_info = basic_info_raw if isinstance(basic_info_raw, dict) else {}
    goals = goals_raw if isinstance(goals_raw, dict) else {}
    preferences = preferences_raw if isinstance(preferences_raw, dict) else {}
    recent_meals = recent_meals_raw if isinstance(recent_meals_raw, list) else []

    reminder_settings = db.notification_settings.find_one({"user_id": user_id}) or {}
    tz_name = str(reminder_settings.get("timezone") or "UTC").strip() or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz_name = "UTC"
        tz = timezone.utc

    now_local = datetime.now(tz)
    local_day_start = datetime(now_local.year, now_local.month, now_local.day, 0, 0, 0, tzinfo=tz)
    local_day_end = local_day_start + timedelta(days=1)
    start_utc = local_day_start.astimezone(timezone.utc)
    end_utc = local_day_end.astimezone(timezone.utc)

    today_meals = list(db.meal_logs.find({"user_id": user_id, "logged_at": {"$gte": start_utc, "$lt": end_utc}}))
    today_calories = sum(_safe_float((m.get("macros") or {}).get("calories_kcal")) for m in today_meals)
    today_meals_count = len(today_meals)

    linked_legacy_ids = {
        m.get("legacy_analysis_id")
        for m in today_meals
        if m.get("legacy_analysis_id") is not None
    }

    legacy_today_cursor = db.collection.find({"user_id": user_id, "created_at": {"$gte": start_utc, "$lt": end_utc}})
    for legacy in legacy_today_cursor:
        legacy_id = legacy.get("_id")
        if legacy_id in linked_legacy_ids:
            continue
        analysis_json = legacy.get("analysis_json") or {}
        tn = analysis_json.get("total_nutrition") or {}
        today_calories += _safe_float(analysis_json.get("calories_kcal", tn.get("calories", 0)))
        today_meals_count += 1

    target_daily_cal = _safe_float(goals.get("daily_calories"), 0)
    if target_daily_cal <= 0:
        target_daily_cal = _safe_float((_get_user_context(user_id) or {}).get("daily_calories"), 2000)
    if target_daily_cal <= 0:
        target_daily_cal = 2000
    remaining_calories = None if target_daily_cal <= 0 else round(max(0.0, target_daily_cal - today_calories), 1)

    now = now_local.astimezone(timezone.utc)
    week_start = _week_start(now)
    plan = db.meal_plans.find_one({"user_id": user_id, "week_start": week_start})
    if not plan:
        plan = db.meal_plans.find_one({"user_id": user_id}, sort=[("week_start", -1)])

    today_iso = now_local.date().isoformat()
    normalized_plan_days = []
    if plan and isinstance(plan.get("days"), list):
        for d in plan.get("days", []):
            date_value = str((d or {}).get("date") or "").strip()
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_value):
                continue
            slots = []
            for s in (d.get("slots") or []):
                slots.append(
                    {
                        "slot": s.get("slot"),
                        "recipe_name": s.get("recipe_name"),
                        "ingredients": (s.get("ingredients") or [])[:6],
                    }
                )
            normalized_plan_days.append({"date": date_value, "slots": slots})

    normalized_plan_days = sorted(normalized_plan_days, key=lambda x: x.get("date") or "")
    upcoming_days = [d for d in normalized_plan_days if (d.get("date") or "") >= today_iso]
    previous_days = [d for d in normalized_plan_days if (d.get("date") or "") < today_iso]
    plan_preview = upcoming_days[:3]
    if len(plan_preview) < 3 and previous_days:
        plan_preview = previous_days[-(3 - len(plan_preview)) :] + plan_preview

    today_plan = next((d for d in normalized_plan_days if d.get("date") == today_iso), None)
    if not today_plan:
        today_plan = {"date": today_iso, "slots": []}

    thirty_start = now - timedelta(days=30)
    recent_30 = list(db.meal_logs.find({"user_id": user_id, "logged_at": {"$gte": thirty_start}}))
    days_logged = len({m.get("logged_at").date().isoformat() for m in recent_30 if isinstance(m.get("logged_at"), datetime)})
    avg_daily_cal = round(sum(_safe_float((m.get("macros") or {}).get("calories_kcal")) for m in recent_30) / max(1, days_logged), 1)

    weights = list(db.weight_logs.find({"user_id": user_id}).sort("date", 1).limit(365))
    weight_delta = None
    if len(weights) >= 2:
        weight_delta = round(_safe_float(weights[-1].get("weight_kg")) - _safe_float(weights[0].get("weight_kg")), 2)

    return {
        "name": user_doc.get("name") or "User",
        "email": user_doc.get("email"),
        "picture": user_doc.get("picture"),
        "current_date_utc": today_iso,
        "current_datetime_utc": now.isoformat(),
        "current_date_local": today_iso,
        "timezone": tz_name,
        "diet_type": preferences.get("diet_type") or "standard_american",
        "goal_type": goals.get("goal_type") or "maintain_weight",
        "daily_calories": target_daily_cal,
        "today_calories": round(today_calories, 1),
        "today_meals_count": today_meals_count,
        "remaining_calories": remaining_calories,
        "budget_per_meal": preferences.get("budget_per_meal"),
        "recent_meals_count": len(recent_meals),
        "days_logged_30": days_logged,
        "avg_daily_calories_30": avg_daily_cal,
        "weight_delta_kg": weight_delta,
        "progress_summary": {
            "days_logged_30": days_logged,
            "avg_daily_calories_30": avg_daily_cal,
            "weight_delta_kg": weight_delta,
        },
        "has_week_plan": bool(plan),
        "today_plan": today_plan,
        "week_plan_preview": plan_preview,
        "basic_info": basic_info,
        "goals": goals,
        "preferences": preferences,
        "recent_meals": recent_meals[:20],
    }


def _append_coach_exchange(session_doc, user_text, assistant_text):
    now = datetime.now(timezone.utc)
    messages = session_doc.get("messages", [])
    messages.extend(
        [
            {"id": str(uuid.uuid4()), "role": "user", "text": user_text, "at": now.isoformat()},
            {"id": str(uuid.uuid4()), "role": "assistant", "text": assistant_text, "at": now.isoformat()},
        ]
    )
    messages = messages[-80:]
    db.chat_sessions.update_one(
        {"_id": session_doc.get("_id")},
        {"$set": {"messages": messages, "updated_at": now}},
    )
    return messages


def _is_date_or_time_question(message):
    text = str(message or "").strip().lower()
    if not text:
        return False

    date_patterns = [
        "today's date",
        "todays date",
        "what is todays date",
        "what is today's date",
        "what's today's date",
        "what day is it",
        "current date",
        "date today",
    ]
    time_patterns = [
        "current time",
        "what time is it",
        "time now",
        "time today",
    ]

    return any(p in text for p in (date_patterns + time_patterns))


@v3_bp.route("/api/v3/coach/history")
def v3_coach_history():
    user_id, err = _auth_guard()
    if err:
        return err
    doc = _coach_history_doc(user_id)
    msgs = doc.get("messages", [])
    return jsonify({"success": True, "messages": msgs[-50:]})


@v3_bp.route("/api/v3/coach/context")
def v3_coach_context():
    user_id, err = _auth_guard()
    if err:
        return err
    return jsonify({"success": True, "context": _coach_context_payload(user_id)})


@v3_bp.route("/api/v3/coach/chat", methods=["POST"])
def v3_coach_chat():
    user_id, err = _auth_guard()
    if err:
        return err
    if not _v3_model:
        return jsonify({"success": False, "error": "ai_unavailable", "message": "Gemini is not available for coach chat."}), 503

    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"success": False, "error": "message_required"}), 400

    context_payload = _coach_context_payload(user_id)
    session_doc = _coach_history_doc(user_id)
    prior_messages = session_doc.get("messages", [])[-10:]

    if _is_date_or_time_question(message):
        now = datetime.now(timezone.utc)
        reply = f"Today's date is {now.strftime('%B %d, %Y')} (UTC)."
        messages = _append_coach_exchange(session_doc, message, reply)
        return jsonify({"success": True, "reply": reply, "messages": messages[-20:]})

    prompt_context = {
        "current_date_utc": context_payload.get("current_date_utc"),
        "current_datetime_utc": context_payload.get("current_datetime_utc"),
        "name": context_payload.get("name"),
        "diet_type": context_payload.get("diet_type"),
        "goal_type": context_payload.get("goal_type"),
        "daily_calories": context_payload.get("daily_calories"),
        "today_calories": context_payload.get("today_calories"),
        "remaining_calories": context_payload.get("remaining_calories"),
        "budget_per_meal": context_payload.get("budget_per_meal"),
        "today_plan": context_payload.get("today_plan"),
        "week_plan_preview": context_payload.get("week_plan_preview"),
        "progress_summary": context_payload.get("progress_summary"),
        "recent_meals": (context_payload.get("recent_meals") or [])[:10],
        "preferences": context_payload.get("preferences") or {},
    }

    prompt = (
        "You are Alimento Coach. Be concise, practical, and direct. "
        "Use all user context and answer exactly what the user asked.\n"
        "Output rules (strict):\n"
        "1) Return Markdown only.\n"
        "2) If question is nutrition/meal/planning: one sentence + 2 or 3 '- ' bullets.\n"
        "3) If question is non-nutrition: one direct sentence, no bullets.\n"
        "4) Total response under 80 words.\n"
        "5) Never invent meals. Use plan recipes only if present in today_plan or week_plan_preview.\n"
        "6) Do not repeat old suggestions unless still relevant to current question.\n"
        "7) If data is missing, state what is missing briefly.\n"
        "8) No filler, no generic fallback text.\n\n"
        f"User context (dynamic runtime): {json.dumps(prompt_context)}\n"
        f"Recent conversation: {json.dumps(prior_messages)}\n"
        f"User question: {message}"
    )
    resp, gem_err = _gemini_generate(prompt)
    reply = (getattr(resp, "text", "") or "").strip() if resp else ""
    if not reply:
        if gem_err == "rate_limit":
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "ai_rate_limited",
                        "message": (
                            "Gemini quota or per-minute limit reached. Wait briefly and try again. "
                            "See https://ai.google.dev/gemini-api/docs/rate-limits"
                        ),
                    }
                ),
                429,
            )
        return jsonify({"success": False, "error": "ai_generation_failed", "message": "Gemini could not produce a coach response."}), 502

    messages = _append_coach_exchange(session_doc, message, reply)

    return jsonify({"success": True, "reply": reply, "messages": messages[-20:]})


def _challenge_member_count(challenge_id):
    return db.challenge_members.count_documents({"challenge_id": challenge_id})


_SOCIAL_GOAL_META = {
    "meal_logging_streak": {"label": "Meal logging", "unit": "days"},
    "hydration_consistency": {"label": "Hydration", "unit": "days"},
    "nutrition_consistency": {"label": "Nutrition consistency", "unit": "days"},
}


def _normalize_challenge_goal(goal):
    token = _normalize_token(goal)
    if token in _SOCIAL_GOAL_META:
        return token
    return "meal_logging_streak"


def _challenge_bounds(challenge):
    now = datetime.now(timezone.utc)
    start_at = challenge.get("start_at")
    end_at = challenge.get("end_at")
    if not isinstance(start_at, datetime):
        start_at = now - timedelta(days=14)
    if not isinstance(end_at, datetime):
        end_at = now
    return start_at, end_at


def _score_meal_days(user_id, start_at, end_at):
    meal_days = set()
    cursor = db.meal_logs.find({"user_id": user_id, "logged_at": {"$gte": start_at, "$lte": end_at}})
    for meal in cursor:
        dt = meal.get("logged_at")
        if isinstance(dt, datetime):
            meal_days.add(dt.date().isoformat())
    return meal_days


def _score_hydration_days(user_id, start_at, end_at):
    start_key = start_at.date().isoformat()
    end_key = end_at.date().isoformat()
    rows = db.hydration_logs.find(
        {
            "user_id": user_id,
            "date": {"$gte": start_key, "$lte": end_key},
        }
    )
    days = set()
    for row in rows:
        date_key = str(row.get("date") or "").strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_key):
            continue
        glasses = _safe_int(row.get("glasses"), 0)
        ml = _safe_int(row.get("ml"), 0)
        if glasses > 0 or ml > 0:
            days.add(date_key)
    return days


def _compute_challenge_score(user_id, challenge):
    start_at, end_at = _challenge_bounds(challenge)
    goal = _normalize_challenge_goal(challenge.get("goal"))
    meal_days = _score_meal_days(user_id, start_at, end_at)

    if goal == "meal_logging_streak":
        return len(meal_days)

    hydration_days = _score_hydration_days(user_id, start_at, end_at)
    if goal == "hydration_consistency":
        return len(hydration_days)

    # nutrition_consistency: count days with either meals or hydration logged.
    return len(meal_days.union(hydration_days))


def _serialize_challenge_for_user(challenge_doc, user_id, creator_cache):
    row = _serialize_oid(challenge_doc)
    start_at, end_at = _challenge_bounds(challenge_doc)
    goal = _normalize_challenge_goal(challenge_doc.get("goal"))

    creator_id = challenge_doc.get("created_by")
    creator_name = "Community"
    if creator_id is not None:
        key = str(creator_id)
        if key not in creator_cache:
            user_doc = db.users.find_one({"_id": creator_id}) or {}
            creator_cache[key] = user_doc.get("name") or "User"
        creator_name = creator_cache.get(key) or "User"

    joined = db.challenge_members.find_one({"challenge_id": challenge_doc.get("_id"), "user_id": user_id}) is not None
    participant_count = _challenge_member_count(challenge_doc.get("_id"))
    your_score = _compute_challenge_score(user_id, challenge_doc) if joined else 0

    total_window_days = max(1, int((end_at.date() - start_at.date()).days) + 1)
    progress_pct = max(0, min(100, int((your_score / total_window_days) * 100)))

    row.update(
        {
            "goal": goal,
            "goal_label": (_SOCIAL_GOAL_META.get(goal) or {}).get("label") or "Meal logging",
            "goal_unit": (_SOCIAL_GOAL_META.get(goal) or {}).get("unit") or "days",
            "start_at": start_at.isoformat(),
            "end_at": end_at.isoformat(),
            "duration_days": total_window_days,
            "participant_count": participant_count,
            "joined": joined,
            "created_by_name": creator_name,
            "is_created_by_you": challenge_doc.get("created_by") == user_id,
            "your_score": your_score,
            "your_progress_pct": progress_pct,
        }
    )

    if isinstance(row.get("created_at"), datetime):
        row["created_at"] = row["created_at"].isoformat()
    if isinstance(row.get("updated_at"), datetime):
        row["updated_at"] = row["updated_at"].isoformat()
    return row


@v3_bp.route("/api/v3/social/challenges", methods=["GET", "POST"])
def v3_social_challenges():
    user_id, err = _auth_guard()
    if err:
        return err

    if request.method == "GET":
        docs = list(db.challenges.find({"is_active": True}).sort("created_at", -1).limit(100))
        creator_cache = {}
        out = [_serialize_challenge_for_user(d, user_id, creator_cache) for d in docs]
        return jsonify({"success": True, "challenges": out})

    body = request.get_json(silent=True) or {}
    name = str(body.get("name") or "").strip() or "Consistency Challenge"
    description = str(body.get("description") or "").strip() or "Log meals daily and stay consistent."
    duration_days = _safe_int(body.get("duration_days", 14), 14)
    duration_days = max(3, min(duration_days, 90))
    goal = _normalize_challenge_goal(body.get("goal"))
    now = datetime.now(timezone.utc)
    doc = {
        "schema_version": 3,
        "name": name,
        "description": description,
        "created_by": user_id,
        "is_active": True,
        "start_at": now,
        "end_at": now + timedelta(days=duration_days),
        "goal": goal,
        "duration_days": duration_days,
        "created_at": now,
        "updated_at": now,
    }
    ins = db.challenges.insert_one(doc)
    db.challenge_members.update_one(
        {"challenge_id": ins.inserted_id, "user_id": user_id},
        {"$set": {"joined_at": now, "score": 0, "schema_version": 3, "updated_at": now}},
        upsert=True,
    )
    doc["_id"] = ins.inserted_id
    out = _serialize_challenge_for_user(doc, user_id, {str(user_id): current_user.name or "User"})
    return jsonify({"success": True, "challenge": out})


@v3_bp.route("/api/v3/social/challenges/<challenge_id>/join", methods=["POST"])
def v3_social_join_challenge(challenge_id):
    user_id, err = _auth_guard()
    if err:
        return err
    try:
        cid = ObjectId(challenge_id)
    except Exception:
        return jsonify({"success": False, "error": "invalid_challenge_id"}), 400

    challenge = db.challenges.find_one({"_id": cid, "is_active": True})
    if not challenge:
        return jsonify({"success": False, "error": "challenge_not_found"}), 404

    now = datetime.now(timezone.utc)
    db.challenge_members.update_one(
        {"challenge_id": cid, "user_id": user_id},
        {"$setOnInsert": {"joined_at": now, "score": 0, "schema_version": 3}},
        upsert=True,
    )
    return jsonify({"success": True})


@v3_bp.route("/api/v3/social/challenges/<challenge_id>/leave", methods=["POST"])
def v3_social_leave_challenge(challenge_id):
    user_id, err = _auth_guard()
    if err:
        return err
    try:
        cid = ObjectId(challenge_id)
    except Exception:
        return jsonify({"success": False, "error": "invalid_challenge_id"}), 400

    challenge = db.challenges.find_one({"_id": cid, "is_active": True})
    if not challenge:
        return jsonify({"success": False, "error": "challenge_not_found"}), 404

    db.challenge_members.delete_one({"challenge_id": cid, "user_id": user_id})
    return jsonify({"success": True})


@v3_bp.route("/api/v3/social/challenges/<challenge_id>/leaderboard")
def v3_social_leaderboard(challenge_id):
    user_id, err = _auth_guard()
    if err:
        return err
    try:
        cid = ObjectId(challenge_id)
    except Exception:
        return jsonify({"success": False, "error": "invalid_challenge_id"}), 400

    challenge = db.challenges.find_one({"_id": cid})
    if not challenge:
        return jsonify({"success": False, "error": "challenge_not_found"}), 404

    members = list(db.challenge_members.find({"challenge_id": cid}))
    if not members:
        return jsonify({"success": True, "leaderboard": []})

    start_at, end_at = _challenge_bounds(challenge)
    goal = _normalize_challenge_goal(challenge.get("goal"))
    leaderboard = []

    for m in members:
        uid = m.get("user_id")
        score = _compute_challenge_score(uid, challenge)
        db.challenge_members.update_one(
            {"challenge_id": cid, "user_id": uid},
            {"$set": {"score": score, "updated_at": datetime.now(timezone.utc)}},
        )
        user_doc = db.users.find_one({"_id": uid}) or {}
        leaderboard.append(
            {
                "user_id": str(uid),
                "name": user_doc.get("name") or "User",
                "picture": user_doc.get("picture"),
                "score": score,
                "goal": goal,
                "goal_label": (_SOCIAL_GOAL_META.get(goal) or {}).get("label") or "Meal logging",
                "goal_unit": (_SOCIAL_GOAL_META.get(goal) or {}).get("unit") or "days",
                "is_you": uid == user_id,
            }
        )

    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    for i, row in enumerate(leaderboard):
        row["rank"] = i + 1
    return jsonify({"success": True, "leaderboard": leaderboard})


@v3_bp.route("/api/v3/settings/reminders", methods=["GET", "POST"])
def v3_settings_reminders():
    user_id, err = _auth_guard()
    if err:
        return err

    if request.method == "GET":
        doc = db.notification_settings.find_one({"user_id": user_id}) or {}
        data = {
            "hydration_enabled": bool(doc.get("hydration_enabled", True)),
            "meal_logging_enabled": bool(doc.get("meal_logging_enabled", True)),
            "weekly_checkin_enabled": bool(doc.get("weekly_checkin_enabled", True)),
            "email_enabled": bool(doc.get("email_enabled", False)),
            "timezone": doc.get("timezone") or "UTC",
        }
        return jsonify({"success": True, "settings": data})

    body = request.get_json(silent=True) or {}
    now = datetime.now(timezone.utc)
    db.notification_settings.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "schema_version": 3,
                "user_id": user_id,
                "hydration_enabled": bool(body.get("hydration_enabled", True)),
                "meal_logging_enabled": bool(body.get("meal_logging_enabled", True)),
                "weekly_checkin_enabled": bool(body.get("weekly_checkin_enabled", True)),
                "email_enabled": bool(body.get("email_enabled", False)),
                "timezone": body.get("timezone") or "UTC",
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    return jsonify({"success": True})


@v3_bp.route("/api/v3/settings/integrations", methods=["GET", "POST"])
def v3_settings_integrations():
    user_id, err = _auth_guard()
    if err:
        return err

    if request.method == "GET":
        rows = list(db.activity_integrations.find({"user_id": user_id}))
        items = []
        for r in rows:
            items.append(
                {
                    "id": str(r.get("_id")),
                    "provider": r.get("provider"),
                    "status": r.get("status", "disconnected"),
                    "last_sync_at": r.get("last_sync_at").isoformat() if isinstance(r.get("last_sync_at"), datetime) else None,
                    "meta": r.get("meta", {}),
                }
            )
        return jsonify({"success": True, "integrations": items})

    body = request.get_json(silent=True) or {}
    provider = (body.get("provider") or "").strip().lower()
    if provider not in {"apple_health", "fitbit", "google_fit", "myfitnesspal"}:
        return jsonify({"success": False, "error": "invalid_provider"}), 400

    action = (body.get("action") or "connect").strip().lower()
    now = datetime.now(timezone.utc)
    status = "connected" if action == "connect" else "disconnected"
    db.activity_integrations.update_one(
        {"user_id": user_id, "provider": provider},
        {
            "$set": {
                "schema_version": 3,
                "user_id": user_id,
                "provider": provider,
                "status": status,
                "last_sync_at": now if status == "connected" else None,
                "meta": {
                    "note": "Integration scaffold enabled. OAuth token exchange can be added next.",
                },
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    return jsonify({"success": True, "provider": provider, "status": status})
