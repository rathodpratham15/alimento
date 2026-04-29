from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, make_response, current_app, session
import google.generativeai as genai
from PIL import Image, ImageEnhance
import requests
from io import BytesIO
import os
import json
from datetime import datetime, timedelta, timezone
import re
import uuid
from werkzeug.utils import secure_filename
from werkzeug.local import LocalProxy
from dotenv import load_dotenv


# Auth-related imports
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from itsdangerous import URLSafeSerializer
from authlib.integrations.flask_client import OAuth
from bson import ObjectId

# MongoDB import
from database import get_db

# Usage tracking import
from usage_tracker import check_limit, track_usage, get_usage_summary, guest_v3_trial_status
from diet_config import score_meal_adherence
from diet_config import (
    DIET_CONFIGURATIONS,
    calculate_bmr,
    calculate_tdee,
    calculate_macro_grams,
    compute_macro_adherence_10pt,
    detect_allergens_from_text,
    portion_feedback,
    goal_specific_advice,
    goal_adjustment_calories,
)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'diet-designer-secret-key-2024')

# VERCEL FIX: Use /tmp directory for uploads (only writable directory)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# VERCEL FIX: Only create directories if not on Vercel
if not os.environ.get('VERCEL'):
    # Local development
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.config['UPLOAD_FOLDER'] = 'uploads'
else:
    # Vercel deployment - use /tmp (only writable directory)
    os.makedirs('/tmp/uploads', exist_ok=True)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables!")
    print("Create a .env file with: GEMINI_API_KEY=your_api_key_here")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully")

# Initialize MongoDB database
db = LocalProxy(get_db)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)

# Serializer for guest session cookie
GUEST_COOKIE_NAME = 'guest_session'
serializer = URLSafeSerializer(app.secret_key, salt='guest-session')


def guest_session_uuid():
    cookie = request.cookies.get(GUEST_COOKIE_NAME)
    if not cookie:
        return None
    try:
        return serializer.loads(cookie)
    except Exception:
        return None


# OAuth client will be configured in auth blueprint

class User(UserMixin):
    def __init__(self, user_doc):
        self.id = str(user_doc.get('_id'))
        self.google_sub = user_doc.get('google_sub')
        self.email = user_doc.get('email')
        self.name = user_doc.get('name')
        self.picture = user_doc.get('picture')

    def get_id(self):
        return self.id
        
    @property
    def is_authenticated(self):
        """Always return True for authenticated users"""
        return True
        
    @property 
    def is_active(self):
        """Always return True - all users are active"""
        return True
        
    @property
    def is_anonymous(self):
        """Always return False for authenticated users"""
        return False


@login_manager.user_loader
def load_user(user_id):
    try:
        user_doc = db.users.find_one({'_id': ObjectId(user_id)})
        if user_doc:
            return User(user_doc)
    except Exception:
        return None


@app.context_processor
def inject_user():
    """Make current_user available in all templates"""
    from flask_login import current_user as _current_user
    return dict(current_user=_current_user)


# Register auth blueprint after User class is defined
from auth import auth_bp, init_oauth
init_oauth(app)  # Initialize OAuth with the Flask app
app.register_blueprint(auth_bp)

# Register profile blueprint
from profile import profile_bp
app.register_blueprint(profile_bp)

# Register v3 features blueprint
from v3_features import v3_bp
app.register_blueprint(v3_bp)


@app.after_request
def _attach_guest_session_cookie(response):
    try:
        if not (current_user and getattr(current_user, "is_authenticated", False)):
            # Re-issue when missing or when present but fails verification (e.g. rotated FLASK_SECRET_KEY).
            if guest_session_uuid() is None:
                gid = str(uuid.uuid4())
                signed = serializer.dumps(gid)
                response.set_cookie(
                    GUEST_COOKIE_NAME,
                    signed,
                    httponly=True,
                    samesite="Lax",
                    secure=bool(os.getenv("PRODUCTION")),
                )
    except Exception:
        pass
    return response


def ensure_guest_cookie(response=None):
    """Ensure guest_session cookie exists for anonymous visitors."""
    cookie = request.cookies.get(GUEST_COOKIE_NAME)
    if not cookie:
        gid = str(uuid.uuid4())
        signed = serializer.dumps(gid)
        if response is None:
            response = make_response()
        response.set_cookie(GUEST_COOKIE_NAME, signed, httponly=True, samesite='Lax', secure=bool(os.getenv('PRODUCTION')))
    return response


def current_identity():
    """Return dict with identity type and id: {'type':'user','id':...} or {'type':'guest','id':...}"""
    if current_user and getattr(current_user, 'is_authenticated', False):
        return {'type': 'user', 'id': current_user.get_id()}
    # else guest
    cookie = request.cookies.get(GUEST_COOKIE_NAME)
    if cookie:
        try:
            gid = serializer.loads(cookie)
            return {'type': 'guest', 'id': gid}
        except Exception:
            # invalid cookie - create a new one
            gid = str(uuid.uuid4())
            return {'type': 'guest', 'id': gid}
    # fallback
    gid = str(uuid.uuid4())
    return {'type': 'guest', 'id': gid}

class DietAnalyzer:
    def __init__(self):
        if GEMINI_API_KEY:
            self.model = genai.GenerativeModel(
                'gemini-2.5-flash-lite',
                generation_config={
                    # Increased to allow large table + payload output
                    "max_output_tokens": 8192,
                    "temperature": 0.7,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
                ]
            )
        else:
            self.model = None
    
    def enhance_image(self, img):
        """Apply basic image enhancements and fix format issues"""
        try:
            # Convert RGBA to RGB if needed
            if img.mode == 'RGBA':
                print("Converting RGBA to RGB for compatibility")
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode == 'P':
                img = img.convert('RGB')
            elif img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            
            # Enhance contrast and brightness
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
            
            # Resize for optimal processing
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            print(f"Image processed: {img.mode} mode, size: {img.size}")
            return img
            
        except Exception as e:
            print(f"Image enhancement error: {e}")
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
    
    def get_diet_info(self, dietary_goal):
        """Get comprehensive diet information"""
        diet_data = {
            "ketogenic": {
                "name": "Ketogenic",
                "rules": "KETO RULES: <20g net carbs daily, 70-80% calories from healthy fats, moderate protein",
                "focus": "Focus on avocados, nuts, olive oil, fatty fish, low-carb vegetables",
                "icon": "🥑",
                "color": "#FF6B35"
            },
            "plant_based_vegan": {
                "name": "Vegan",
                "rules": "VEGAN RULES: No animal products (meat, dairy, eggs, honey)",
                "focus": "Focus on legumes, nuts, seeds, whole grains, fruits, vegetables",
                "icon": "🌱",
                "color": "#4CAF50"
            },
            "vegetarian": {
                "name": "Vegetarian",
                "rules": "VEGETARIAN: No meat/fish. Eggs and dairy allowed.",
                "focus": "Plant-forward with eggs/dairy for protein. Whole grains, legumes.",
                "icon": "🥦",
                "color": "#8BC34A"
            },
            "paleo": {
                "name": "Paleo",
                "rules": "PALEO RULES: No processed foods, grains, legumes, dairy, refined sugar",
                "focus": "Focus on grass-fed meats, wild fish, eggs, vegetables, fruits, nuts",
                "icon": "🥩",
                "color": "#D84315"
            },
            "mediterranean": {
                "name": "Mediterranean",
                "rules": "MEDITERRANEAN: High in olive oil, fish, vegetables, whole grains, moderate wine",
                "focus": "Focus on olive oil, fish, vegetables, legumes, whole grains, herbs",
                "icon": "🫒",
                "color": "#1976D2"
            },
            "low_carb": {
                "name": "Low Carb",
                "rules": "LOW-CARB: <100g carbs daily, emphasis on protein and healthy fats",
                "focus": "Focus on lean proteins, healthy fats, non-starchy vegetables",
                "icon": "⚖️",
                "color": "#9C27B0"
            },
            "intermittent_fasting_18_6": {
                "name": "Intermittent Fasting 18:6",
                "rules": "FASTING 18:6: Eat only within 6-hour window. Hydrate during fast.",
                "focus": "Nutrient density during eating window",
                "icon": "⏳",
                "color": "#607D8B"
            },
             "intermittent_fasting_16_8": {
                "name": "Intermittent Fasting 16:8",
                "rules": "FASTING 16:8: Eat only within 8-hour window.",
                "focus": "Balanced meals during window",
                "icon": "⏳",
                "color": "#607D8B"
            },
            "standard_american": {
                 "name": "Standard American",
                 "rules": "STANDARD: Balanced macronutrients (50% carb, 20% protein, 30% fat).",
                 "focus": "Portion control, whole foods, limiting processed sugars.",
                 "icon": "🍽️",
                 "color": "#607D8B"
            },
            "flexitarian": {
                 "name": "Flexitarian",
                 "rules": "FLEXITARIAN: Mostly plant-based, occasional meat permitted.",
                 "focus": "Increase plants, reduce meat frequency/portion.",
                 "icon": "🥗",
                 "color": "#8BC34A"
            },
            "pescatarian": {
                 "name": "Pescatarian",
                 "rules": "PESCATARIAN: Vegetarian + Fish/Seafood.",
                 "focus": "Omega-3s from fish, plant proteins, vegetables.",
                 "icon": "🐟",
                 "color": "#03A9F4"
            },
            "dash_diet": {
                 "name": "DASH Diet",
                 "rules": "DASH: Low sodium (<1500-2300mg), high potassium/magnesium.",
                 "focus": "Lower blood pressure: Fruits, veggies, low-fat dairy.",
                 "icon": "🧂",
                 "color": "#00BCD4"
            },
            "gluten_free": {
                 "name": "Gluten-Free",
                 "rules": "GF RULES: Strictly NO wheat, barley, rye.",
                 "focus": "Avoid hidden gluten. Use rice, corn, quinoa, potatoes.",
                 "icon": "🌾",
                 "color": "#FFC107"
            },
            "low_fodmap": {
                 "name": "Low FODMAP",
                 "rules": "LOW-FODMAP: Avoid high-FODMAP carbs (onions, garlic, wheat, certain fruits).",
                 "focus": "Digestive relief. Eat rice, potatoes, carrots, spinach, maple syrup.",
                 "icon": "🥝",
                 "color": "#8D6E63"
            },
            "whole30": {
                 "name": "Whole30",
                 "rules": "WHOLE30: No sugar, alcohol, grains, legumes, dairy for 30 days.",
                 "focus": "Reset. Meat, seafood, eggs, veggies, fruit, natural fats only.",
                 "icon": "🍎",
                 "color": "#D32F2F"
            },
            "anti_inflammatory": {
                 "name": "Anti-Inflammatory",
                 "rules": "ANTI-INFLAMMATORY: High omega-3s, antioxidants. Low sugar/processed.",
                 "focus": "Berries, fatty fish, leafy greens, olive oil, turmeric.",
                 "icon": "🫐",
                 "color": "#E91E63"
            }
        }
        
        return diet_data.get(dietary_goal, {
            "name": "Healthy",
            "rules": "HEALTHY EATING: Balanced nutrition, whole foods",
            "focus": "Focus on nutrient-dense whole foods",
            "icon": "🍎",
            "color": "#607D8B"
        })
    
    def analyze_meal(self, image_path, dietary_goal, user_preferences=""):
        """Analyze meal with comprehensive AI assessment"""
        if not self.model:
            return {"error": "Gemini API not configured. Please set GEMINI_API_KEY in .env file"}
        
        try:
            print(f"Loading image from: {image_path}")
            
            # Load and enhance image
            img = Image.open(image_path)
            print(f"Original image: {img.mode} mode, size: {img.size}")
            
            img = self.enhance_image(img)
            
            # VERCEL FIX: Save processed image to /tmp
            processed_path = image_path.replace('.', '_processed.')
            if not processed_path.lower().endswith(('.jpg', '.jpeg')):
                processed_path = processed_path + '.jpg'
            
            img.save(processed_path, 'JPEG', quality=90)
            print(f"Processed image saved: {processed_path}")
            
            diet_info = self.get_diet_info(dietary_goal)
            
            # Enhanced analysis prompt without markdown formatting
            prompt = f"""COMPREHENSIVE MEAL ANALYSIS FOR {diet_info['name'].upper()} DIET {diet_info['icon']}

Please analyze this meal image and provide a detailed, well-structured analysis using clean text formatting (NO MARKDOWN SYMBOLS like ** or *):

MEAL IDENTIFICATION:
List all visible food items with estimated portions and cooking methods.

NUTRITIONAL ESTIMATION:
Provide estimates for:
• Total Calories: [number] kcal
• Carbohydrates: [number]g (including fiber)
• Protein: [number]g 
• Fat: [number]g
• Key vitamins/minerals present
• Sodium level: [Low/Medium/High]

DIET COMPATIBILITY SCORE: [X]/10
{diet_info['rules']}

POSITIVE ASPECTS:
• What makes this meal good for {dietary_goal} diet
• Health benefits identified
• Nutritionally strong points

AREAS FOR IMPROVEMENT:
• What doesn't align with {dietary_goal} diet
• Specific concerns or issues
• Missing nutrients

PERSONALIZED RECOMMENDATIONS:
{diet_info['focus']}
1. Ingredient Modifications: Specific swaps to make
2. Portion Adjustments: What to increase/decrease
3. Preparation Changes: Better cooking methods
4. Additions: What to add to make it more {dietary_goal}-friendly

OVERALL HEALTH SCORE: [X]/10
Explanation of why this score was given.

PERSONALIZED ADVICE:
{f'Based on your preferences: {user_preferences}' if user_preferences else 'General recommendations for optimal nutrition'}

SUMMARY:
One paragraph summary of the meal's suitability for {dietary_goal} diet and key takeaways.

Please be specific with numbers, practical with suggestions, and format the response clearly with the section headers shown above. Use NO markdown symbols like asterisks or underscores."""

            # Generate analysis
            response = self.model.generate_content([prompt, img])
            
            if response.text:
                # Create thumbnail for storage (Base64)
                img_thumb = img.copy()
                img_thumb.thumbnail((600, 600))
                buffered = BytesIO()
                img_thumb.save(buffered, format="JPEG", quality=85)
                import base64
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Store analysis data
                analysis_data = {
                    "timestamp": datetime.now().isoformat(),
                    "dietary_goal": dietary_goal,
                    "diet_info": diet_info,
                    "analysis": response.text,
                    "user_preferences": user_preferences,
                    "image_path": processed_path,
                    "image_base64": img_base64 # Added Base64 for persistent storage
                }
                
                print("Analysis completed successfully")
                return {"success": True, "analysis": response.text, "data": analysis_data}
            else:
                return {"error": "AI returned empty response. Please try again."}
                
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}

    def analyze_meal_with_profile(self, image_path, user_context, meal_context: str = ""):
        """Analyze meal using full user profile and return structured JSON.
        user_context keys expected: age, gender, weight_kg, height_cm, activity_level, diet_type,
        daily_calorie_target, protein_target, carb_target, fat_target, allergies, health_conditions, restrictions
        """
        if not self.model:
            return {"error": "Gemini API not configured. Please set GEMINI_API_KEY in .env file"}

        try:
            img = Image.open(image_path)
            img = self.enhance_image(img)
            processed_path = image_path.replace('.', '_processed.')
            if not processed_path.lower().endswith(('.jpg', '.jpeg')):
                processed_path = processed_path + '.jpg'
            img.save(processed_path, 'JPEG', quality=90)

            # Build prompt to produce Markdown + DATA_PAYLOAD tail
            uc = user_context or {}
            system_profile = f"""
You are a professional nutritionist. Analyze the MEAL IMAGE with the USER PROFILE below and output
ONLY: (1) a clean Markdown report and (2) a fenced JSON code block labeled DATA_PAYLOAD.

USER PROFILE:
- Age: {uc.get('age','N/A')}, Gender: {uc.get('gender','N/A')}
- Weight: {uc.get('weight_kg','N/A')}kg, Height: {uc.get('height_cm','N/A')}cm
- Activity: {uc.get('activity_level','N/A')}
- Diet Type: {uc.get('diet_type','N/A')}
- Daily Targets: {uc.get('daily_calorie_target','N/A')} cal, {uc.get('protein_target','N/A')}g protein, {uc.get('carb_target','N/A')}g carbs, {uc.get('fat_target','N/A')}g fat
- Allergies: {', '.join(uc.get('allergies',[]) or []) or 'None'}
- Health Conditions: {', '.join(uc.get('health_conditions',[]) or []) or 'None'}
- Food Restrictions: {', '.join(uc.get('restrictions',[]) or []) or 'None'}
- Meal Context: {meal_context or 'general'}

STRICT OUTPUT CONTRACT:
- Markdown sections (exact order):
  1) # <Diet Type> Diet Analysis
  2) **Meal Breakdown** table (Item | Portion | Method | Notes)
  3) **Macros & Key Nutrients** table (Total Calories | Carbs (g) | Protein (g) | Fat (g) | Fiber (g) | Sodium (mg))
     - Add sodium/fiber notes when applicable
  4) **Diet Compatibility Score** bold (e.g., **Score: 5/10**)
  5) **Positives**
  6) **Areas for Improvement**
  7) **Personalized Recommendations** with three bold sublists:
     - **Ingredient Swaps**, **Portion Tweaks**, **Cooking Methods** (3–5 bullets each)
  8) **Overall Health Score** (1–2 sentences)
- Do NOT include dates/timestamps anywhere in the markdown.
- After the markdown, append a fenced code block named DATA_PAYLOAD with keys:
  {"meal_identification","diet_type","calories_kcal","carbs_g","protein_g","fat_g","fiber_g","sodium_mg","adherence_score","flags","top_violations","top_suggestions"}
- No extra commentary; keep lines under ~100 chars.
"""

            response = self.model.generate_content([system_profile, img])
            # Robust text extraction for multi-part responses
            raw = ""
            try:
                if hasattr(response, 'text') and response.text:
                    raw = response.text
                elif getattr(response, 'candidates', None):
                    parts = getattr(response.candidates[0].content, 'parts', [])
                    texts = []
                    for p in parts:
                        t = getattr(p, 'text', None)
                        if t:
                            texts.append(t)
                    raw = "\n".join(texts)
            except Exception:
                raw = ""

            # Extract DATA_PAYLOAD and markdown
            import re
            md = raw or ""
            payload = {}
            m = re.search(r"```\s*DATA_PAYLOAD[\w\s]*\n([\s\S]*?)```", raw or "")
            if m:
                json_part = m.group(1)
                try:
                    payload = json.loads(json_part)
                except Exception:
                    payload = {}
                md = (raw[:m.start()]).strip()
            else:
                # Fallback: any fenced JSON code block
                m2 = re.search(r"```\s*(?:json)?\s*\n(\{[\s\S]*?\})\s*```", raw or "")
                if m2:
                    try:
                        payload = json.loads(m2.group(1))
                        md = (raw[:m2.start()]).strip()
                    except Exception:
                        payload = {}
                else:
                    # Fallback: last JSON-like object in text
                    start = (raw or '').rfind('{')
                    end = (raw or '').rfind('}')
                    if start != -1 and end != -1 and end > start:
                        try:
                            payload = json.loads((raw or '')[start:end+1])
                            md = (raw[:start]).strip()
                        except Exception:
                            payload = {}
            
            # Remove any remaining fenced code blocks (e.g., unlabeled JSON) from the visible markdown section
            md = re.sub(r"```[\s\S]*?```", "", md).strip()
            # Remove standalone ISO-like date lines if any slipped in
            md = "\n".join([ln for ln in md.splitlines() if not re.match(r"^\s*20\d{2}[-/].*", ln)]).strip()

            # Normalize payload keys for downstream logic
            def _num(x):
                try:
                    return float(x)
                except Exception:
                    return None
            if payload:
                # Map alt shapes to flat keys
                tn = payload.get('total_nutrition') or {}
                if 'calories_kcal' not in payload:
                    if 'calories' in payload:
                        payload['calories_kcal'] = _num(payload.get('calories'))
                    elif 'calories' in tn:
                        payload['calories_kcal'] = _num(tn.get('calories'))
                for k_src, k_dst in [('carbs','carbs_g'), ('protein','protein_g'), ('fat','fat_g'), ('fiber','fiber_g'), ('sodium','sodium_mg')]:
                    if k_dst not in payload:
                        if k_src in payload:
                            payload[k_dst] = _num(payload.get(k_src))
                        elif k_src in tn:
                            payload[k_dst] = _num(tn.get(k_src))
                # Ensure numbers are numeric
                for key in ['calories_kcal','carbs_g','protein_g','fat_g','fiber_g','sodium_mg']:
                    if key in payload:
                        payload[key] = _num(payload.get(key))

            def _looks_structured(_md: str, _payload: dict) -> bool:
                has_tables = ('|' in (_md or '')) and ('Meal Breakdown' in (_md or '') or 'Macros' in (_md or ''))
                has_core = bool(_payload) and any(k in _payload for k in ['calories_kcal','carbs_g','protein_g','fat_g'])
                return has_tables and has_core

            # If the first pass doesn't satisfy structure, attempt a repair pass
            if not _looks_structured(md, payload):
                try:
                    repair_prompt = f"""
You are a formatter. Take the ANALYSIS CONTENT below and output EXACTLY:
1) Clean Markdown with these sections in order:
   # <Diet Type> Diet Analysis
   Meal Breakdown (as a Markdown table with headers: Item | Portion | Method | Notes)
   Macros & Key Nutrients (as a Markdown table with headers: Nutrient | Amount)
   Diet Compatibility Score: X/10
   Positives (bullets)
   Areas for Improvement (bullets)
   Personalized Recommendations with bold subheads (Ingredient Swaps, Portion Tweaks, Cooking Methods)
   Overall Health Score (1–2 sentences)
2) Then append a fenced code block named DATA_PAYLOAD containing JSON with keys:
   {{"meal_identification","diet_type","calories_kcal","carbs_g","protein_g","fat_g","fiber_g","sodium_mg","adherence_score","flags","top_violations","top_suggestions"}}
No extra commentary. Keep lines < 100 chars. Do not include any other code blocks.

USER PROFILE SUMMARY:
- Diet Type: {uc.get('diet_type','N/A')}, Goal: {uc.get('goal_type','maintain_weight')}, Activity: {uc.get('activity_level','N/A')}
- Allergies: {', '.join(uc.get('allergies',[]) or []) or 'None'} | Restrictions: {', '.join(uc.get('restrictions',[]) or []) or 'None'}
- Meal Context: {meal_context or 'general'}

ANALYSIS CONTENT:
{raw}
"""
                    response2 = self.model.generate_content(repair_prompt)
                    raw2 = ''
                    if hasattr(response2, 'text') and response2.text:
                        raw2 = response2.text
                    elif getattr(response2, 'candidates', None):
                        parts = getattr(response2.candidates[0].content, 'parts', [])
                        raw2 = "\n".join([getattr(p, 'text', '') for p in parts if getattr(p, 'text', '')])

                    # Parse repaired
                    md2 = raw2 or ""
                    payload2 = {}
                    m3 = re.search(r"```\s*DATA_PAYLOAD[\w\s]*\n([\s\S]*?)```", raw2 or "")
                    if m3:
                        try:
                            payload2 = json.loads(m3.group(1))
                        except Exception:
                            payload2 = {}
                        md2 = (raw2[:m3.start()]).strip()
                    else:
                        m4 = re.search(r"```\s*(?:json)?\s*\n(\{[\s\S]*?\})\s*```", raw2 or "")
                        if m4:
                            try:
                                payload2 = json.loads(m4.group(1))
                                md2 = (raw2[:m4.start()]).strip()
                            except Exception:
                                payload2 = {}
                    md2 = re.sub(r"```[\s\S]*?```", "", md2).strip()

                    # Normalize payload2 keys
                    if payload2:
                        tn2 = payload2.get('total_nutrition') or {}
                        if 'calories_kcal' not in payload2:
                            if 'calories' in payload2:
                                payload2['calories_kcal'] = _num(payload2.get('calories'))
                            elif 'calories' in tn2:
                                payload2['calories_kcal'] = _num(tn2.get('calories'))
                        for k_src, k_dst in [('carbs','carbs_g'), ('protein','protein_g'), ('fat','fat_g'), ('fiber','fiber_g'), ('sodium','sodium_mg')]:
                            if k_dst not in payload2:
                                if k_src in payload2:
                                    payload2[k_dst] = _num(payload2.get(k_src))
                                elif k_src in tn2:
                                    payload2[k_dst] = _num(tn2.get(k_src))
                        for key in ['calories_kcal','carbs_g','protein_g','fat_g','fiber_g','sodium_mg']:
                            if key in payload2:
                                payload2[key] = _num(payload2.get(key))

                    # If repair looks good, replace
                    if _looks_structured(md2, payload2):
                        md, payload = md2, payload2
                except Exception:
                    pass

            if not _looks_structured(md, payload):
                return {"success": False, "error": "structured_markdown_missing", "raw_text": raw, "processed_image": processed_path}

            # Create thumbnail for storage (Base64)
            img_thumb = img.copy()
            img_thumb.thumbnail((600, 600))
            buffered = BytesIO()
            img_thumb.save(buffered, format="JPEG", quality=85)
            import base64
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return {"success": True, "markdown": md, "data_payload": payload, "processed_image": processed_path, "image_base64": img_base64}

        except Exception as e:
            print(f"Profile analysis error: {str(e)}")
            return {"error": f"Profile analysis failed: {str(e)}"}
    
    def extract_nutrition_data(self, analysis_text):
        """Extract key numerical data for display cards"""
        nutrition_data = {}
        
        try:
            # Extract calories
            calories_match = re.search(r'calories?:?\s*(\d+)', analysis_text, re.IGNORECASE)
            if calories_match:
                nutrition_data['calories'] = int(calories_match.group(1))
            
            # Extract macronutrients
            macros = {
                'carbs': r'carbohydrates?:?\s*(\d+)g',
                'protein': r'protein:?\s*(\d+)g', 
                'fat': r'fat:?\s*(\d+)g'
            }
            
            for macro, pattern in macros.items():
                match = re.search(pattern, analysis_text, re.IGNORECASE)
                if match:
                    nutrition_data[macro] = int(match.group(1))
            
            # Extract scores
            compatibility_match = re.search(r'compatibility.*?(\d+)/10', analysis_text, re.IGNORECASE)
            if compatibility_match:
                nutrition_data['compatibility_score'] = int(compatibility_match.group(1))
                
            health_match = re.search(r'health.*?score.*?(\d+)/10', analysis_text, re.IGNORECASE)
            if health_match:
                nutrition_data['health_score'] = int(health_match.group(1))

            # Extract sodium level if present (Low/Medium/High)
            sodium_match = re.search(r'sodium\s+level:\s*(low|medium|high)', analysis_text, re.IGNORECASE)
            if sodium_match:
                nutrition_data['sodium_level'] = sodium_match.group(1).lower()
            
            print(f"Extracted nutrition data: {nutrition_data}")
            
        except Exception as e:
            print(f"Data extraction warning: {e}")
        
        return nutrition_data

# Initialize analyzer
analyzer = DietAnalyzer()

@app.route('/')
def index():
    """Main page with meal analysis form"""
    # Ensure guest cookie for anonymous users
    resp = make_response(render_template('index.html'))
    ensure_guest_cookie(resp)
    return resp

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle meal analysis requests with usage limits"""
    try:
        # Check usage limits first
        limit_check = check_limit('analyses')
        if not limit_check['allowed']:
            return jsonify({
                'error': 'limit_exceeded',
                'feature': 'analyze',
                'limit': limit_check['limit'],
                'current': limit_check['current'],
                'user_type': limit_check['user_type'],
                'message': f"Daily limit reached ({limit_check['current']}/{limit_check['limit']}). {'Sign in for higher limits.' if limit_check['user_type'] == 'guest' else 'Try again tomorrow.'}"
            }), 429  # Too Many Requests
        
        image_path = None
        
        # Handle file upload - VERCEL COMPATIBLE
        if 'image_file' in request.files and request.files['image_file'].filename:
            file = request.files['image_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = os.path.splitext(filename)[0]
                filename = f"{timestamp}_{filename_base}.jpg"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                file.save(image_path)
                print(f"File uploaded: {image_path}")
        
        # Handle URL input
        elif request.form.get('image_url'):
            try:
                response = requests.get(request.form.get('image_url'), timeout=15)
                response.raise_for_status()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"url_image_{timestamp}.jpg"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                img = Image.open(BytesIO(response.content))
                img = analyzer.enhance_image(img)
                img.save(image_path, 'JPEG', quality=90)
                print(f"URL image processed and saved: {image_path}")
                
            except Exception as e:
                return jsonify({"error": f"Failed to download image: {str(e)}"})
        
        if not image_path:
            return jsonify({"error": "Please provide an image file or URL"})
        
        # Get form data
        diet_goal = request.form.get('diet_goal', 'keto')
        user_preferences = request.form.get('user_preferences', '').strip()
        
        print(f"Analyzing for {diet_goal} diet")
        
        # Analyze meal
        result = analyzer.analyze_meal(image_path, diet_goal, user_preferences)
        
        if result.get("success"):
            # Only save to database if user is signed in
            db_save_result = None
            if current_user and getattr(current_user, 'is_authenticated', False):
                db_save_result = save_to_history(result["data"], None)
            
            # Track usage after successful analysis
            track_usage('analyses')
            
            # Compute adherence score to selected diet
            extracted = analyzer.extract_nutrition_data(result["analysis"])
            adherence = None
            try:
                from profile import db as _db
                if current_user and getattr(current_user, 'is_authenticated', False):
                    prefs = _db.diet_preferences.find_one({'user_id': ObjectId(current_user.id)}) or {}
                    diet_slug = prefs.get('diet_type') or 'standard_american'
                else:
                    diet_slug = 'standard_american'
                adherence = score_meal_adherence({
                    'carbs': extracted.get('carbs'),
                    'protein': extracted.get('protein'),
                    'fat': extracted.get('fat'),
                    'sodium_mg': None,
                    'sodium_level': extracted.get('sodium_level')
                }, diet_slug)
            except Exception as _:
                adherence = None

            return jsonify({
                "success": True,
                "analysis": result["analysis"],
                "chart_url": None,
                "nutrition_data": extracted,
                "diet_info": analyzer.get_diet_info(diet_goal),
                "adherence": adherence,
                "database_id": db_save_result.get("id") if db_save_result and db_save_result.get("success") else None
            })
        else:
            return jsonify(result)
            
    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"})

def _history_sort_datetime(row):
    """Parse unified history row timestamp for sorting (newest first)."""
    t = row.get("timestamp")
    if isinstance(t, datetime):
        dt = t
    elif isinstance(t, str):
        try:
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.min.replace(tzinfo=timezone.utc)
    else:
        dt = datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _history_row_json_safe(row):
    """Top-level BSON types -> JSON-serializable (legacy docs may include datetime)."""
    out = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, ObjectId):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _meal_log_to_history_item(doc, fallback_diet="standard_american"):
    """Map a meal_logs document to the shape expected by history.html."""
    macros = doc.get("macros") or {}
    cal = macros.get("calories_kcal")
    pg = macros.get("protein_g")
    cg = macros.get("carbs_g")
    fg = macros.get("fat_g")
    meal_name = doc.get("meal_name") or "Meal"
    src = doc.get("source") or "manual"
    notes = (doc.get("notes") or "").strip()
    raw = (doc.get("raw_input") or "").strip()
    diet = doc.get("diet_type") or fallback_diet
    logged = doc.get("logged_at") or doc.get("created_at")
    if isinstance(logged, datetime):
        if logged.tzinfo:
            ts = logged.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        else:
            ts = logged.isoformat() + "Z"
    else:
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    macro_line = ""
    if cal is not None:
        try:
            macro_line = f"{float(cal):.0f} kcal"
            macro_line += f" · P {float(pg or 0):.0f}g · C {float(cg or 0):.0f}g · F {float(fg or 0):.0f}g"
        except (TypeError, ValueError):
            macro_line = str(cal) + " kcal"

    desc_parts = []
    if raw and raw != "manual_entry":
        desc_parts.append(f"**Description:** {raw}")
    if notes:
        desc_parts.append(f"**Notes:** {notes}")
    analysis_md = f"## {meal_name}\n\n**Source:** {src}  \n**Macros:** {macro_line}\n\n"
    if desc_parts:
        analysis_md += "\n\n".join(desc_parts)

    aj = {
        "meal_identification": {"name": meal_name},
        "calories_kcal": cal,
    }
    if cal is not None:
        aj["nutritional_estimation"] = {"calories": cal}

    return {
        "_id": str(doc["_id"]),
        "history_kind": "v3",
        "timestamp": ts,
        "analysis": analysis_md,
        "dietary_goal": diet,
        "analysis_json": aj,
        "image_base64": doc.get("image_base64"),
        "image_path": doc.get("image_path"),
        "meal_type": doc.get("meal_type"),
        "source": src,
    }


def _build_unified_history(uid):
    """Legacy photo analyses (db.collection) + v3 meal_logs, newest first, capped."""
    prefs = db.diet_preferences.find_one({"user_id": uid}) or {}
    fallback_diet = prefs.get("diet_type") or "standard_american"

    history = []
    cursor = db.collection.find({"user_id": uid}).sort("created_at", -1).limit(200)
    for doc in cursor:
        row = dict(doc)
        row["_id"] = str(row["_id"])
        if "user_id" in row and row["user_id"] is not None:
            row["user_id"] = str(row["user_id"])
        ca = row.get("created_at")
        if isinstance(ca, datetime):
            row["timestamp"] = (
                ca.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                if ca.tzinfo
                else ca.isoformat() + "Z"
            )
        row["history_kind"] = "legacy"
        history.append(_history_row_json_safe(row))

    ml_cursor = db.meal_logs.find({"user_id": uid}).sort("logged_at", -1).limit(500)
    for doc in ml_cursor:
        history.append(_meal_log_to_history_item(doc, fallback_diet))

    history.sort(key=_history_sort_datetime, reverse=True)
    return history[:400]


@app.route('/history')
def history():
    """Analysis + meal log timeline (client loads via /api/history)."""
    try:
        is_guest = not (current_user and getattr(current_user, "is_authenticated", False))
        return render_template("history.html", history=[], is_guest=is_guest)
    except Exception as e:
        print(f"History error: {e}")
        return render_template("history.html", history=[], is_guest=True)

@app.route('/api/history')
def api_history():
    """Unified timeline: legacy analyses and v3 meal_logs, newest first."""
    try:
        if not (current_user and getattr(current_user, "is_authenticated", False)):
            return jsonify(
                {"success": True, "history": [], "count": 0, "is_guest": True}
            )

        uid = ObjectId(current_user.id)
        history = _build_unified_history(uid)

        return jsonify(
            {
                "success": True,
                "history": history,
                "count": len(history),
                "is_guest": False,
                "user_id": current_user.id,
            }
        )
    except Exception as e:
        print(f"API History error: {e}")
        return jsonify(
            {"success": False, "error": str(e), "history": [], "is_guest": True}
        )

@app.route('/clear-history', methods=['POST'])
@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear legacy analyses and v3 meal logs for the current user."""
    try:
        if not (current_user and getattr(current_user, "is_authenticated", False)):
            return jsonify({"success": False, "error": "Must be signed in to clear history"})

        uid = ObjectId(current_user.id)
        res_legacy = db.collection.delete_many({"user_id": uid})
        res_v3 = db.meal_logs.delete_many({"user_id": uid})
        return jsonify(
            {
                "success": True,
                "deleted_count": res_legacy.deleted_count + res_v3.deleted_count,
                "deleted_legacy": res_legacy.deleted_count,
                "deleted_meal_logs": res_v3.deleted_count,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/delete-analysis/<analysis_id>', methods=['POST'])
def delete_analysis(analysis_id):
    """Delete specific analysis - SIGNED IN USERS ONLY"""
    try:
        # Only allow signed-in users to delete analyses
        if not (current_user and getattr(current_user, 'is_authenticated', False)):
            return jsonify({"success": False, "error": "Must be signed in to delete analyses"})
        
        # Delete only if owned by current user
        obj_id = ObjectId(analysis_id)
        res = db.collection.delete_one({'_id': obj_id, 'user_id': ObjectId(current_user.id)})

        if res.deleted_count > 0:
            return jsonify({"success": True, "message": "Analysis deleted successfully"})
        else:
            return jsonify({"success": False, "error": "Analysis not found or not owned by you"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/stats')
def stats():
    """Get database statistics"""
    try:
        stats = db.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/debug-auth')
def debug_auth():
    """Debug authentication status"""
    debug_info = {
        "current_user_exists": current_user is not None,
        "is_authenticated": getattr(current_user, 'is_authenticated', False),
        "user_id": getattr(current_user, 'id', None),
        "user_email": getattr(current_user, 'email', None),
        "user_name": getattr(current_user, 'name', None),
        "user_picture": getattr(current_user, 'picture', None),
        "session_keys": list(session.keys()) if session else [],
    }
    return jsonify(debug_info)

@app.route('/api/me')
def api_me():
    """Get current user info"""
    if current_user and getattr(current_user, 'is_authenticated', False):
        return jsonify({
            'authenticated': True,
            'user': {
                'id': current_user.id,
                'email': current_user.email,
                'name': current_user.name,
                'picture': current_user.picture
            }
        })
    else:
        # Ensure guest cookie exists
        cookie = request.cookies.get(GUEST_COOKIE_NAME)
        resp = make_response(jsonify({'authenticated': False, 'user': None}))
        if not cookie:
            gid = str(uuid.uuid4())
            signed = serializer.dumps(gid)
            resp.set_cookie(GUEST_COOKIE_NAME, signed, httponly=True, samesite='Lax', secure=bool(os.getenv('PRODUCTION')))
        return resp


@app.route('/api/usage')
def api_usage():
    """Get current usage status and limits"""
    from usage_tracker import get_current_scope, get_user_type, LIMITS
    
    user_type = get_user_type()
    scope = get_current_scope()
    usage_summary = get_usage_summary(scope)
    
    limits = LIMITS[user_type]
    
    # Calculate usage percentages and warnings
    usage_status = {}
    for feature, limit in limits.items():
        current = usage_summary.get(feature, 0)
        usage_status[feature] = {
            'current': current,
            'limit': limit,
            'percentage': (current / limit * 100) if limit > 0 else 0,
            'near_limit': current >= (limit * 0.8) if limit > 0 else False,
            'at_limit': current >= limit if limit > 0 else False
        }
    
    return jsonify({
        'user_type': user_type,
        'scope': scope,
        'usage': usage_status,
        'raw_usage': usage_summary
    })


@app.route('/api/analyze-with-profile', methods=['POST'])
def api_analyze_with_profile():
    """Analyze meal with full user profile context. Requires image and uses current user's saved data.
    If guest, falls back to standard analysis prompt without profile-specific targets.
    """
    try:
        # Validate image input
        image_path = None
        if 'image_file' in request.files and request.files['image_file'].filename:
            file = request.files['image_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = os.path.splitext(filename)[0]
                filename = f"{timestamp}_{filename_base}.jpg"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)
        elif request.form.get('image_url'):
            response = requests.get(request.form.get('image_url'), timeout=15)
            response.raise_for_status()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"url_image_{timestamp}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = Image.open(BytesIO(response.content))
            img = analyzer.enhance_image(img)
            img.save(image_path, 'JPEG', quality=90)
        else:
            return jsonify({"success": False, "error": "Please provide an image"}), 400

        # Build user context if authenticated
        meal_context = request.form.get('meal_context', '')
        user_context = {}
        if current_user and getattr(current_user, 'is_authenticated', False):
            # Load user-specific data
            prefs = db.diet_preferences.find_one({'user_id': ObjectId(current_user.id)}) or {}
            prof = db.user_profiles.find_one({'user_id': ObjectId(current_user.id)}) or {}
            goals = db.nutrition_goals.find_one({'user_id': ObjectId(current_user.id)}) or {}
            user_context = {
                'age': prof.get('age'),
                'gender': prof.get('biological_sex'),
                'weight_kg': prof.get('weight_kg'),
                'height_cm': prof.get('height_cm'),
                'activity_level': prof.get('activity_level'),
                'medications': prof.get('medications'),
                'supplements': prof.get('supplements', []),
                'diet_type': (prefs.get('diet_type') or 'standard_american'),
                'allergies': prefs.get('allergies', []),
                'health_conditions': prof.get('health_conditions', []),
                'daily_calorie_target': goals.get('daily_calories'),
                'protein_target': goals.get('protein_grams'),
                'carb_target': goals.get('carbs_grams'),
                'fat_target': goals.get('fat_grams'),
                'restrictions': prefs.get('food_restrictions', []),
                'goal_type': goals.get('goal_type') or 'maintain_weight',
                'living_situation': prefs.get('living_situation'),
                'meal_prep_preference': prefs.get('meal_prep_preference'),
                'cooking_skill': prefs.get('cooking_skill'),
                'budget_per_meal': prefs.get('budget_per_meal'),
                'class_schedule': prefs.get('class_schedule'),
            }

        result = analyzer.analyze_meal_with_profile(image_path, user_context, meal_context)
        if not result.get('success'):
            return jsonify(result), 500

        # If model returned markdown + payload, use that; else use previous fallback path
        markdown = result.get('markdown')
        payload = result.get('data_payload') or {}

        analysis_text = None
        if markdown:
            analysis_text = markdown
        structured = {}
        if payload:
            structured = payload

        if not analysis_text:
            # Enforce table-based markdown output only; remove legacy narrative fallback
            return jsonify({
                'success': False,
                'error': 'structured_markdown_missing',
                'message': 'The AI did not return the expected table-based markdown. Please try again.'
            }), 502

        # Fallback: parse macros from Markdown if payload missing
        def parse_macros_from_markdown(md_text: str):
            import re
            if not md_text:
                return {}
            t = md_text
            # Pipe-table pattern
            m = re.search(r"\|\s*Total\s*Calories\s*\|[\s\S]*?\n\|[\-:\s\|]+\n\|\s*(?P<cal>\d+(?:\.\d+)?)\s*\|\s*(?P<carb>\d+(?:\.\d+)?)\s*\|\s*(?P<pro>\d+(?:\.\d+)?)\s*\|\s*(?P<fat>\d+(?:\.\d+)?)\s*\|\s*(?P<fiber>\d+(?:\.\d+)?)\s*\|\s*(?P<sod>\d+(?:\.\d+)?)", t, re.IGNORECASE)
            if m:
                return {
                    'calories_kcal': float(m.group('cal')),
                    'carbs_g': float(m.group('carb')),
                    'protein_g': float(m.group('pro')),
                    'fat_g': float(m.group('fat')),
                    'fiber_g': float(m.group('fiber')),
                    'sodium_mg': float(m.group('sod')),
                }
            # Loose text fallback
            def pick(rx):
                mm = re.search(rx, t, re.IGNORECASE)
                return float(mm.group(1)) if mm else None
            cal = pick(r"Total\s*Calories\D+(\d+(?:\.\d+)?)")
            carb = pick(r"Carbs\s*\(g\)\D+(\d+(?:\.\d+)?)")
            pro = pick(r"Protein\s*\(g\)\D+(\d+(?:\.\d+)?)")
            fat = pick(r"Fat\s*\(g\)\D+(\d+(?:\.\d+)?)")
            fiber = pick(r"Fiber\s*\(g\)\D+(\d+(?:\.\d+)?)")
            sod = pick(r"Sodium\s*\(mg\)\D+(\d+(?:\.\d+)?)")
            got = {k:v for k,v in [('calories_kcal',cal),('carbs_g',carb),('protein_g',pro),('fat_g',fat),('fiber_g',fiber),('sodium_mg',sod)] if v is not None}
            return got

        if analysis_text and (not structured or not any(structured.get(k) for k in ['calories_kcal','carbs_g','protein_g','fat_g'])):
            parsed = parse_macros_from_markdown(analysis_text)
            if parsed:
                structured.update(parsed)

        # Compute personalization using configs and user profile (if available)
        personalization = {}
        try:
            diet_slug = user_context.get('diet_type', 'standard_american')
            daily_target_kcal = user_context.get('daily_calorie_target')
            # Fallback: compute daily target if missing using BMR/TDEE and goal adjustment
            if not daily_target_kcal:
                try:
                    if all(user_context.get(k) for k in ['weight_kg','height_cm','age','gender','activity_level']):
                        bmr = calculate_bmr(float(user_context['weight_kg']), float(user_context['height_cm']), int(user_context['age']), user_context['gender'])
                        tdee = calculate_tdee(bmr, user_context['activity_level'])
                        adj = goal_adjustment_calories(user_context.get('goal_type') or 'maintain_weight')
                        daily_target_kcal = max(1200, int(tdee + adj))
                except Exception:
                    daily_target_kcal = None
            if structured:
                macro_score = compute_macro_adherence_10pt(
                    structured.get('calories_kcal'),
                    structured.get('carbs_g'),
                    structured.get('protein_g'),
                    structured.get('fat_g'),
                    diet_slug,
                )
                portion_msg = portion_feedback(structured.get('calories_kcal'), daily_target_kcal, meal_context)
            else:
                macro_score = {"score": None, "explanation": "No structured macros"}
                portion_msg = portion_feedback(None, daily_target_kcal, meal_context)
            allergens = detect_allergens_from_text(analysis_text, user_context.get('allergies', []))
            # Dynamic goal tips based on deviations and sodium
            goal_tips = goal_specific_advice(user_context.get('goal_type'))
            try:
                tips_dynamic = []
                if macro_score and macro_score.get('explanation') and 'carbs off' in macro_score.get('explanation').lower():
                    tips_dynamic.append("Reduce refined carbs; add more non-starchy vegetables.")
                if macro_score and macro_score.get('explanation') and 'protein off' in macro_score.get('explanation').lower():
                    tips_dynamic.append("Add a lean protein portion to balance macros.")
                if structured.get('sodium_mg') and structured['sodium_mg'] > 1500:
                    tips_dynamic.append("Choose fresh items and limit salty seasonings to reduce sodium.")
                if structured.get('calories_kcal') and daily_target_kcal:
                    pct = structured['calories_kcal'] / max(1, daily_target_kcal)
                    if pct > 0.6:
                        tips_dynamic.append("Since this meal is large, keep other meals lighter today.")
                if tips_dynamic:
                    goal_tips = list(dict.fromkeys(goal_tips + tips_dynamic))
            except Exception:
                pass
            cfg = DIET_CONFIGURATIONS.get(diet_slug) or {}
            limits = cfg.get('daily_limits') or cfg.get('daily_targets')
            personalization = {
                'macro_adherence': macro_score,
                'portion_advice': portion_msg,
                'allergen_matches': allergens,
                'goal_tips': goal_tips,
                'diet_limits': limits,
            }
        except Exception as _:
            personalization = {}

        # Attach ownership and save result
        save_payload = {
            'timestamp': datetime.now().isoformat(),
            'dietary_goal': user_context.get('diet_type', 'standard_american'),
            'analysis': analysis_text,
            'analysis_json': structured,
            'personalization': personalization,
            'image_path': result.get('processed_image'),
            'image_base64': result.get('image_base64'),
            'meal_context': meal_context
        }
        db_result = None
        if current_user and getattr(current_user, 'is_authenticated', False):
            db_result = save_to_history(save_payload, None)

        v3_meal_id = None
        if current_user and getattr(current_user, 'is_authenticated', False):
            try:
                uid = ObjectId(current_user.id)

                def to_num(value, default=0.0):
                    try:
                        return float(value)
                    except Exception:
                        return float(default)

                now_utc = datetime.now(timezone.utc)
                legacy_analysis_id = None
                if db_result and db_result.get('success') and db_result.get('id'):
                    try:
                        legacy_analysis_id = ObjectId(str(db_result.get('id')))
                    except Exception:
                        legacy_analysis_id = None

                raw_input = request.form.get('image_url') or (request.files.get('image_file').filename if request.files.get('image_file') else 'image_upload')
                meal_doc = {
                    'schema_version': 3,
                    'user_id': uid,
                    'source': 'analyze_with_profile',
                    'meal_name': structured.get('meal_name') or 'Meal from analysis',
                    'notes': meal_context or '',
                    'diet_type': user_context.get('diet_type') or 'standard_american',
                    'meal_type': meal_context or 'unspecified',
                    'macros': {
                        'calories_kcal': to_num(structured.get('calories_kcal')),
                        'protein_g': to_num(structured.get('protein_g')),
                        'carbs_g': to_num(structured.get('carbs_g')),
                        'fat_g': to_num(structured.get('fat_g')),
                        'fiber_g': to_num(structured.get('fiber_g')),
                        'sodium_mg': to_num(structured.get('sodium_mg')),
                    },
                    'image_base64': result.get('image_base64'),
                    'barcode': None,
                    'raw_input': raw_input,
                    'metadata': {
                        'analysis_json': structured,
                        'legacy_analysis_id': str(legacy_analysis_id) if legacy_analysis_id else None,
                    },
                    'logged_at': now_utc,
                    'created_at': now_utc,
                    'updated_at': now_utc,
                }
                if legacy_analysis_id:
                    meal_doc['legacy_analysis_id'] = legacy_analysis_id
                    db.meal_logs.update_one(
                        {'legacy_analysis_id': legacy_analysis_id},
                        {'$setOnInsert': meal_doc},
                        upsert=True,
                    )
                    existing = db.meal_logs.find_one({'legacy_analysis_id': legacy_analysis_id}, {'_id': 1})
                    if existing and existing.get('_id'):
                        v3_meal_id = str(existing.get('_id'))
                else:
                    ins = db.meal_logs.insert_one(meal_doc)
                    v3_meal_id = str(ins.inserted_id)
            except Exception as sync_err:
                print(f"v3 meal sync warning: {sync_err}")

        return jsonify({
            'success': True,
            'structured': structured,
            'analysis': analysis_text,
            'personalization': personalization,
            'database_id': db_result.get('id') if db_result and db_result.get('success') else None,
            'v3_meal_id': v3_meal_id,
        })

    except Exception as e:
        print(f"analyze-with-profile error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
        
        






@app.route('/ping')
def ping():
    """Lightweight keep-warm endpoint without DB access."""
    return jsonify({
        'ok': True,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }), 200


@app.route('/health')
def health_check():
    """App health without forcing DB initialization."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'database': 'not_checked'
    }), 200


@app.route('/health/db')
def health_db_check():
    """Explicit database health check endpoint."""
    try:
        manager = get_db()
        if not manager.client:
            return jsonify({
                'status': 'unhealthy',
                'database': 'disconnected',
                'error': 'Database client not initialized'
            }), 500

        manager.client.admin.command('ping')
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'database': 'connected'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'database': 'error',
            'error': str(e)
        }), 500

@app.route('/privacy')
def privacy():
    return render_template('legal/privacy.html')

@app.route('/terms')
def terms():
    return render_template('legal/terms.html')


@app.route('/fix-users', methods=['POST'])
def fix_users():
    """Fix corrupted users with null google_sub"""
    try:
        # Find users with null or missing google_sub
        bad_users = list(db.users.find({'$or': [{'google_sub': None}, {'google_sub': {'$exists': False}}]}))
        
        if not bad_users:
            return jsonify({"success": True, "message": "No bad users found"})
        
        # Delete bad users (they'll be recreated properly on next login)
        result = db.users.delete_many({'$or': [{'google_sub': None}, {'google_sub': {'$exists': False}}]})
        
        return jsonify({
            "success": True, 
            "message": f"Deleted {result.deleted_count} corrupted users. They will be recreated properly on next login.",
            "deleted_users": [{'email': u.get('email'), '_id': str(u['_id'])} for u in bad_users]
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })



def save_to_history(analysis_data, chart_path):
    """Save analysis to MongoDB database"""
    try:
        if chart_path:
            analysis_data['chart_path'] = chart_path
        # Attach ownership based on current identity
        ident = current_identity()
        if ident['type'] == 'user':
            analysis_data['user_id'] = ObjectId(ident['id'])
            analysis_data['guest_session_id'] = None
        else:
            analysis_data['guest_session_id'] = ident['id']
            analysis_data['user_id'] = None

        result = db.save_analysis(analysis_data)
        if result["success"]:
            print(f"Analysis saved to MongoDB with ID: {result['id']}")
        else:
            print(f"Database save error: {result['error']}")
        
        return result
            
    except Exception as e:
        print(f"History save error: {e}")
        return {"success": False, "error": str(e)}



@app.route('/delete-account', methods=['POST'])
def delete_account():
    """Permanently delete user account and all associated data"""
    try:
        if not (current_user and getattr(current_user, 'is_authenticated', False)):
            return jsonify({'success': False, 'error': 'auth_required'}), 401
            
        uid = ObjectId(current_user.id)
        uid_str = str(uid)

        created_challenges = list(db.challenges.find({'created_by': uid}, {'_id': 1}))
        created_challenge_ids = [c.get('_id') for c in created_challenges if c.get('_id') is not None]

        db.collection.delete_many({'user_id': uid})
        db.meal_logs.delete_many({'user_id': uid})
        db.food_items.delete_many({'user_id': uid})
        db.recipes.delete_many({'user_id': uid})
        db.meal_plans.delete_many({'user_id': uid})
        db.grocery_lists.delete_many({'user_id': uid})
        db.weight_logs.delete_many({'user_id': uid})
        db.chat_sessions.delete_many({'user_id': uid})
        db.hydration_logs.delete_many({'user_id': uid})

        db.challenge_members.delete_many({'user_id': uid})
        if created_challenge_ids:
            db.challenge_members.delete_many({'challenge_id': {'$in': created_challenge_ids}})
        db.challenges.delete_many({'created_by': uid})

        db.activity_integrations.delete_many({'user_id': uid})
        db.notification_settings.delete_many({'user_id': uid})

        db.user_profiles.delete_many({'user_id': uid})
        db.diet_preferences.delete_many({'user_id': uid})
        db.nutrition_goals.delete_many({'user_id': uid})

        db.logins.delete_many({'user_id': uid})
        db.share_links.delete_many({'user_id': uid})
        db.usage.delete_many({'scope': f'user:{uid_str}'})

        db.barcode_cache.delete_many({'created_by': uid})
        db.migration_state.delete_many({'$or': [
            {'name': f'analysis_history_to_meal_logs:{uid_str}'},
            {'name': {'$regex': f':{re.escape(uid_str)}$'}},
        ]})

        db.users.delete_one({'_id': uid})
        
        logout_user()
        flash('Your account has been permanently deleted.', 'info')
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Delete account error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

    
def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

# --- Public Share Route ---
@app.route('/share/<analysis_id>')
def share_analysis(analysis_id):
    """Publicly shareable analysis view (No Auth Required)"""
    try:
        # Fetch analysis
        oid = ObjectId(analysis_id)
        doc = db.collection.find_one({'_id': oid})
        
        if not doc:
            return "Analysis not found", 404
            
        # Parse content for template
        aj = doc.get('analysis_json') or {}
        
        # Robust Name Extraction
        meal_name = 'Unknown Meal'
        mid = aj.get('meal_identification')
        if isinstance(mid, str):
            meal_name = mid
        elif isinstance(mid, dict):
            meal_name = mid.get('name') or 'Unknown Meal'
            
        # Robust Macro Extraction
        macros = {
            'calories': 0, 'carbs_g': 0, 'protein_g': 0, 'fat_g': 0
        }
        ne = aj.get('nutritional_estimation') or {} # Legacy
        tn = aj.get('total_nutrition') or {} # New
        
        # Flatten macros logic
        if 'calories_kcal' in aj: macros['calories'] = aj['calories_kcal']
        elif 'calories' in ne: macros['calories'] = ne['calories']
        elif 'calories' in tn: macros['calories'] = tn['calories']
        
        for k in ['carbs_g', 'protein_g', 'fat_g']:
            base = k.replace('_g', '')
            if k in aj: macros[k] = aj[k]
            elif base in ne: macros[k] = ne[base]
            elif base in tn: macros[k] = tn[base]
            
        # Prepare context
        analysis_data = {
            'meal_name': meal_name,
            'image_base64': doc.get('image_base64'),
            'dietary_goal': doc.get('dietary_goal') or 'general',
            'timestamp_str': doc.get('created_at').strftime('%B %d, %Y • %I:%M %p') if doc.get('created_at') else 'Unknown Date',
            'timestamp_iso': (doc.get('created_at').isoformat() + 'Z') if doc.get('created_at') else '', # Pass ISO for client conversion
            'raw_markdown': doc.get('analysis', '').replace('```DATA_PAYLOAD', '<!--').replace('```', ''), # Strip payload
            'macros': macros
        }
        
        return render_template('share.html', analysis=analysis_data)
        
    except Exception as e:
        print(f"Share error: {e}")
        return "Invalid Link", 404

# Favicon and icon routes for comprehensive device support
@app.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    return app.send_static_file('icon32.png')

@app.route('/apple-touch-icon.png')
def apple_touch_icon():
    """Serve Apple touch icon"""
    return app.send_static_file('icon256.png')

@app.route('/android-chrome-192x192.png')
def android_chrome_192():
    """Serve Android Chrome icon 192x192"""
    return app.send_static_file('icon256.png')

@app.route('/android-chrome-512x512.png')
def android_chrome_512():
    """Serve Android Chrome icon 512x512"""
    return app.send_static_file('icon512.png')

@app.route('/favicon-16x16.png')
def favicon_16():
    """Serve 16x16 favicon"""
    return app.send_static_file('icon16.png')

@app.route('/favicon-32x32.png')
def favicon_32():
    """Serve 32x32 favicon"""
    return app.send_static_file('icon32.png')

@app.route('/safari-pinned-tab.svg')
def safari_pinned_tab():
    """Serve Safari pinned tab icon"""
    return app.send_static_file('icon512.png')

@app.route('/manifest.json')
def manifest():
    """Serve web app manifest"""
    return app.send_static_file('manifest.json')

@app.route('/browserconfig.xml')
def browserconfig():
    """Serve browser config for Windows"""
    return app.send_static_file('browserconfig.xml')

# Catch-all for missing PNG favicons - serve appropriate icon
@app.route('/mstile-<size>.png')
def mstile_fallback(size):
    """Serve appropriate icon for missing MS tile icons"""
    if size in ['70x70', '150x150']:
        return app.send_static_file('icon128.png')
    elif size in ['310x310', '310x150']:
        return app.send_static_file('icon256.png')
    else:
        return app.send_static_file('icon128.png')

@app.route('/dashboard')
def dashboard():
    """Dashboard — signed-in users see full history; guests see trial meals only."""
    return render_template('dashboard.html')


def _parse_client_offset(raw_offset, default=0):
    try:
        val = int(raw_offset)
    except (ValueError, TypeError):
        val = default
    return max(-840, min(840, val))


def _resolve_dashboard_day(offset_min, date_str=None):
    utc_now = datetime.now(timezone.utc)
    local_now = utc_now - timedelta(minutes=offset_min)
    local_today = local_now.date()

    if date_str:
        try:
            target_date = datetime.strptime(str(date_str), '%Y-%m-%d').date()
        except ValueError:
            target_date = local_today
    else:
        target_date = local_today

    local_start = datetime(target_date.year, target_date.month, target_date.day)
    start = local_start + timedelta(minutes=offset_min)
    end = start + timedelta(days=1)
    return target_date, start, end, (target_date == local_today), local_today

@app.route('/api/dashboard/today')
def dashboard_today():
    """Today's metrics — full data for signed-in users; guest trial meals for anonymous visitors."""
    try:
        offset_min = _parse_client_offset(request.args.get('offset', 0), default=0)
        target_date, start, end, is_today, local_today = _resolve_dashboard_day(offset_min, request.args.get('date'))

        if current_user and getattr(current_user, 'is_authenticated', False):
            uid = ObjectId(current_user.id)

            meals = list(db.collection.find({'user_id': uid, 'created_at': {'$gte': start, '$lt': end}}).sort('created_at', 1))
            for m in meals:
                m['_id'] = str(m['_id'])
                m['user_id'] = str(m['user_id']) if m.get('user_id') else None

            v3_meals_raw = list(db.meal_logs.find({'user_id': uid, 'logged_at': {'$gte': start, '$lt': end}}).sort('logged_at', 1))
            for m in v3_meals_raw:
                macros = m.get('macros') or {}
                m['_id'] = str(m['_id'])
                m['created_at'] = m.get('logged_at') or m.get('created_at')
                m['analysis_json'] = {
                    'calories_kcal': macros.get('calories_kcal'),
                    'protein_g': macros.get('protein_g'),
                    'carbs_g': macros.get('carbs_g'),
                    'fat_g': macros.get('fat_g'),
                    'meal_name': m.get('meal_name'),
                    'meal_identification': m.get('meal_name'),
                    'source': 'v3',
                }

            meals = sorted(meals + v3_meals_raw, key=lambda x: x.get('created_at') or x.get('logged_at') or '')

            total_cal = 0.0
            carbs_g = 0.0
            protein_g = 0.0
            fat_g = 0.0
            adherence_scores = []
            for m in meals:
                sj = m.get('analysis_json') or {}
                if 'calories_kcal' in sj:
                    total_cal += float(sj.get('calories_kcal') or 0)
                    carbs_g += float(sj.get('carbs_g') or 0)
                    protein_g += float(sj.get('protein_g') or 0)
                    fat_g += float(sj.get('fat_g') or 0)
                elif sj.get('total_nutrition'):
                    tn = sj['total_nutrition']
                    total_cal += float(tn.get('calories') or 0)
                    carbs_g += float(tn.get('carbs') or 0)
                    protein_g += float(tn.get('protein') or 0)
                    fat_g += float(tn.get('fat') or 0)
                pers = m.get('personalization') or {}
                ms = pers.get('macro_adherence', {}).get('score')
                if ms is not None:
                    adherence_scores.append(float(ms))

            avg_adherence = round(sum(adherence_scores)/len(adherence_scores), 1) if adherence_scores else None

            prefs = db.diet_preferences.find_one({'user_id': uid}) or {}
            prof = db.user_profiles.find_one({'user_id': uid}) or {}
            goals = db.nutrition_goals.find_one({'user_id': uid}) or {}
            diet_slug = prefs.get('diet_type') or 'standard_american'
            daily_target = goals.get('daily_calories')
            if not daily_target:
                try:
                    bmr = calculate_bmr(float(prof.get('weight_kg') or 0), float(prof.get('height_cm') or 0), int(prof.get('age') or 0), prof.get('biological_sex') or 'female')
                    tdee = calculate_tdee(bmr, prof.get('activity_level') or 'sedentary')
                    daily_target = max(1200, int(tdee + goal_adjustment_calories(goals.get('goal_type') or 'maintain_weight')))
                except Exception:
                    daily_target = None

            today_key = target_date.strftime('%Y-%m-%d')
            hyd = db.hydration_logs.find_one({'user_id': uid, 'date': today_key}) or {'glasses': 0, 'ml': 0}

            return jsonify({
                'success': True,
                'guest_mode': False,
                'diet_type': diet_slug,
                'totals': {
                    'calories': round(total_cal, 1),
                    'carbs_g': round(carbs_g, 1),
                    'protein_g': round(protein_g, 1),
                    'fat_g': round(fat_g, 1),
                },
                'targets': {
                    'calories': daily_target,
                    'macros_g': calculate_macro_grams(daily_target, diet_slug) if daily_target else None
                },
                'adherence_avg': avg_adherence,
                'meals': [
                    {
                        'id': m['_id'],
                        'ts': (m['created_at'].isoformat() + 'Z') if hasattr(m.get('created_at'), 'isoformat') else (m.get('created_at') or m.get('timestamp')),
                        'analysis_json': m.get('analysis_json'),
                        'personalization': m.get('personalization'),
                        'image_path': m.get('image_path'),
                        'image_base64': m.get('image_base64'),
                        'source_kind': 'v3' if (m.get('analysis_json') or {}).get('source') == 'v3' else 'legacy',
                        'meal_type': m.get('meal_type'),
                        'raw_input': m.get('raw_input'),
                    } for m in meals
                ],
                'hydration': {
                    'glasses': hyd.get('glasses', 0),
                    'ml': hyd.get('ml', 0)
                },
                'selected_date': target_date.isoformat(),
                'is_today': is_today,
                'local_today': local_today.isoformat(),
            })

        gid = guest_session_uuid()
        if not gid:
            return jsonify({'success': False, 'error': 'auth_required'}), 401

        meals = []
        v3_meals_raw = list(db.meal_logs.find({'guest_session_id': gid, 'logged_at': {'$gte': start, '$lt': end}}).sort('logged_at', 1))
        for m in v3_meals_raw:
            macros = m.get('macros') or {}
            m['_id'] = str(m['_id'])
            m['created_at'] = m.get('logged_at') or m.get('created_at')
            m['analysis_json'] = {
                'calories_kcal': macros.get('calories_kcal'),
                'protein_g': macros.get('protein_g'),
                'carbs_g': macros.get('carbs_g'),
                'fat_g': macros.get('fat_g'),
                'meal_name': m.get('meal_name'),
                'meal_identification': m.get('meal_name'),
                'source': 'v3',
            }

        total_cal = 0.0
        carbs_g = 0.0
        protein_g = 0.0
        fat_g = 0.0
        adherence_scores = []
        for m in v3_meals_raw:
            sj = m.get('analysis_json') or {}
            if 'calories_kcal' in sj:
                total_cal += float(sj.get('calories_kcal') or 0)
                carbs_g += float(sj.get('carbs_g') or 0)
                protein_g += float(sj.get('protein_g') or 0)
                fat_g += float(sj.get('fat_g') or 0)
            pers = m.get('personalization') or {}
            ms = pers.get('macro_adherence', {}).get('score')
            if ms is not None:
                adherence_scores.append(float(ms))

        avg_adherence = round(sum(adherence_scores)/len(adherence_scores), 1) if adherence_scores else None
        diet_slug = 'standard_american'
        daily_target = 2000

        return jsonify({
            'success': True,
            'guest_mode': True,
            'guest_trial': guest_v3_trial_status(gid),
            'diet_type': diet_slug,
            'totals': {
                'calories': round(total_cal, 1),
                'carbs_g': round(carbs_g, 1),
                'protein_g': round(protein_g, 1),
                'fat_g': round(fat_g, 1),
            },
            'targets': {
                'calories': daily_target,
                'macros_g': calculate_macro_grams(daily_target, diet_slug),
            },
            'adherence_avg': avg_adherence,
            'meals': [
                {
                    'id': m['_id'],
                    'ts': (m['created_at'].isoformat() + 'Z') if hasattr(m.get('created_at'), 'isoformat') else (m.get('created_at') or m.get('timestamp')),
                    'analysis_json': m.get('analysis_json'),
                    'personalization': m.get('personalization'),
                    'image_path': m.get('image_path'),
                    'image_base64': m.get('image_base64'),
                    'source_kind': 'v3',
                    'meal_type': m.get('meal_type'),
                    'raw_input': m.get('raw_input'),
                } for m in v3_meals_raw
            ],
            'hydration': {'glasses': 0, 'ml': 0},
            'selected_date': target_date.isoformat(),
            'is_today': is_today,
            'local_today': local_today.isoformat(),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dashboard/hydration', methods=['POST'])
def dashboard_hydration():
    if not (current_user and getattr(current_user, 'is_authenticated', False)):
        return jsonify({'success': False, 'error': 'auth_required'}), 401
    try:
        from datetime import datetime, timezone
        uid = ObjectId(current_user.id)
        payload = request.get_json() or {}
        add_glasses = int(payload.get('add_glasses', 1))
        add_ml = int(payload.get('add_ml', 250))
        
        # Get client timezone offset (minutes)
        try:
            offset_min = int(payload.get('offset', 300))
        except (ValueError, TypeError):
            offset_min = 300

        # Calculate "User's Today" based on offset
        utc_now = datetime.now(timezone.utc)
        local_now = utc_now - timedelta(minutes=offset_min)
        
        # Calculate hydration for today (Local Day)
        today_key = local_now.strftime('%Y-%m-%d')
        existing = db.hydration_logs.find_one({'user_id': uid, 'date': today_key})
        if existing:
            db.hydration_logs.update_one({'_id': existing['_id']}, {'$inc': {'glasses': add_glasses, 'ml': add_ml}})
        else:
            db.hydration_logs.insert_one({'user_id': uid, 'date': today_key, 'glasses': add_glasses, 'ml': add_ml})
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard/insights')
def dashboard_insights():
    """Generate smart insights for the signed-in user.
    Uses Gemini if available, with a heuristic fallback.
    """
    if not (current_user and getattr(current_user, 'is_authenticated', False)):
        return jsonify({'success': False, 'error': 'auth_required'}), 401
    try:
        from datetime import datetime, timezone, timedelta
        uid = ObjectId(current_user.id)

        offset_min = _parse_client_offset(request.args.get('offset', 0), default=0)
        target_date, start, _end, _is_today, _local_today = _resolve_dashboard_day(offset_min, request.args.get('date'))
        
        # Week start is 6 days before 'Today'
        week_start = start - timedelta(days=6)

        # Fetch last 7 days
        meals = list(db.collection.find({'user_id': uid, 'created_at': {'$gte': week_start, '$lt': start + timedelta(days=1)}}).sort('created_at', 1))

        # Aggregate by day
        daily = {}
        for m in meals:
            dt_utc = m.get('created_at')
            # Shift UTC timestamp to User's Local Time for correct bucketing
            if dt_utc:
                dt_local = dt_utc - timedelta(minutes=offset_min)
                key = dt_local.date().isoformat()
            else:
                key = start.date().isoformat()
                
            d = daily.setdefault(key, {'calories': 0.0, 'carbs_g': 0.0, 'protein_g': 0.0, 'fat_g': 0.0, 'count': 0})
            sj = m.get('analysis_json') or {}
            if 'calories_kcal' in sj:
                d['calories'] += float(sj.get('calories_kcal') or 0)
                d['carbs_g'] += float(sj.get('carbs_g') or 0)
                d['protein_g'] += float(sj.get('protein_g') or 0)
                d['fat_g'] += float(sj.get('fat_g') or 0)
            elif sj.get('total_nutrition'):
                tn = sj['total_nutrition']
                d['calories'] += float(tn.get('calories') or 0)
                d['carbs_g'] += float(tn.get('carbs') or 0)
                d['protein_g'] += float(tn.get('protein') or 0)
                d['fat_g'] += float(tn.get('fat') or 0)
            d['count'] += 1

        # Targets
        prefs = db.diet_preferences.find_one({'user_id': uid}) or {}
        prof = db.user_profiles.find_one({'user_id': uid}) or {}
        goals = db.nutrition_goals.find_one({'user_id': uid}) or {}
        diet_slug = prefs.get('diet_type') or 'standard_american'
        daily_target = goals.get('daily_calories')
        if not daily_target:
            try:
                bmr = calculate_bmr(float(prof.get('weight_kg') or 0), float(prof.get('height_cm') or 0), int(prof.get('age') or 0), prof.get('biological_sex') or 'female')
                tdee = calculate_tdee(bmr, prof.get('activity_level') or 'sedentary')
                daily_target = max(1200, int(tdee + goal_adjustment_calories(goals.get('goal_type') or 'maintain_weight')))
            except Exception:
                daily_target = None
        macro_targets = calculate_macro_grams(daily_target, diet_slug) if daily_target else None

        # Build a concise context
        today_key = target_date.isoformat()
        today = daily.get(today_key, {'calories': 0, 'carbs_g': 0, 'protein_g': 0, 'fat_g': 0, 'count': 0})

        insights = []
        used_ai = False
        try:
            # Use Gemini if available
            if analyzer.model:
                summary = {
                    'diet_type': diet_slug,
                    'target_calories': daily_target,
                    'target_macros_g': macro_targets,
                    'today': today,
                    'week': daily
                }
                prompt = (
                    "You are a nutrition coach. Given this JSON summary of a student's meals (today and last 7 days), "
                    "generate 3-5 short, actionable insights (one line each). Focus on protein adequacy, fiber/veg, "
                    "sodium if high, balance vs diet type, and practical next steps for the rest of the day. "
                    "Return plain text bullets starting with '-' only.\n\nSUMMARY:\n" + json.dumps(summary)
                )
                resp = analyzer.model.generate_content(prompt)
                text = ''
                if hasattr(resp, 'text') and resp.text:
                    text = resp.text
                elif getattr(resp, 'candidates', None):
                    parts = getattr(resp.candidates[0].content, 'parts', [])
                    text = "\n".join([getattr(p, 'text', '') for p in parts if getattr(p, 'text', '')])
                for line in text.splitlines():
                    line = line.strip().lstrip('-').strip()
                    if line:
                        insights.append(line)
                used_ai = True
        except Exception:
            used_ai = False

        # Heuristic fallback
        if not insights:
            if daily_target:
                diff = today['calories'] - daily_target
                if diff < -200:
                    insights.append("You're under calories so far—consider a protein-rich meal later.")
                elif diff > 200:
                    insights.append("Calories are high today—choose lighter, high-fiber options next meal.")
            if macro_targets:
                if today['protein_g'] < macro_targets['protein'] * 0.5:
                    insights.append("Protein looks low—add eggs, yogurt, tofu, or lean meat.")
                if today['carbs_g'] > macro_targets['carbs'] * 0.9:
                    insights.append("Carbs are nearing target—prefer non-starchy veggies for volume.")
                if today['fat_g'] > macro_targets['fat'] * 1.1:
                    insights.append("Fat slightly high—keep dressings and oils moderate next meal.")
            if not insights:
                insights.append("Keep prioritizing whole foods and hydrate regularly today.")

        return jsonify({'success': True, 'insights': insights, 'used_ai': used_ai, 'selected_date': target_date.isoformat()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Diet Designer Web App Starting...")
    print("Python version:", __import__('sys').version)
    print("Flask version:", __import__('flask').__version__)
    
    if GEMINI_API_KEY:
        print("Gemini API key configured")
    else:
        print("Gemini API key missing - create .env file")
    
    print("MongoDB connection is lazy and initializes on first DB request")
    
    print("Starting server at: http://localhost:5001")
    print("Access from mobile: http://your-ip:5001")
    print("MongoDB Atlas integration enabled")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
