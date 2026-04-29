# usage_tracker.py - Daily usage limits and tracking
from datetime import datetime, timezone
from database import get_db
from flask_login import current_user
from flask import current_app, request
import uuid

# Lifetime guest trial for V3 meal logging (see guest_v3_trial_* helpers).
GUEST_V3_TRIAL_LIMIT = 3

# Daily limits configuration
LIMITS = {
    'guest': {
        # Increased free user limit from 3 -> 10
        'analyses': 10,
        'ai_search': 0,   # Not available for guests
        'share_links': 0  # Not available for guests
    },
    'user': {
        # Increased signed-in user limit from 25 -> 100
        'analyses': 100,
        'ai_search': 10,
        'share_links': 5  # Active links at any time
    }
}

def get_current_scope():
    """Get the current user's scope for usage tracking"""
    if current_user and getattr(current_user, 'is_authenticated', False):
        return f"user:{current_user.id}"
    
    # For guests, use session cookie (same signing key as app / v3 guest APIs)
    from itsdangerous import URLSafeSerializer

    serializer = URLSafeSerializer(current_app.secret_key, salt='guest-session')
    cookie = request.cookies.get('guest_session')
    
    if cookie:
        try:
            gid = serializer.loads(cookie)
            return f"guest:{gid}"
        except:
            # Invalid cookie, create new guest ID
            gid = str(uuid.uuid4())
            return f"guest:{gid}"
    
    # No cookie, create new guest ID
    gid = str(uuid.uuid4())
    return f"guest:{gid}"

def get_user_type():
    """Get current user type: 'guest' or 'user'"""
    if current_user and getattr(current_user, 'is_authenticated', False):
        return 'user'
    return 'guest'

def get_today_date():
    """Get today's date in YYYYMMDD format (UTC)"""
    return datetime.now(timezone.utc).strftime('%Y%m%d')

def get_usage_count(scope, feature, date=None):
    """Get current usage count for a specific feature"""
    if date is None:
        date = get_today_date()
    
    db = get_db()
    if not db.client:
        return 0
    
    try:
        usage_doc = db.usage.find_one({'scope': scope, 'date': date})
        if usage_doc and 'counters' in usage_doc and feature in usage_doc['counters']:
            return usage_doc['counters'][feature]
        return 0
    except Exception as e:
        print(f"Error getting usage count: {e}")
        return 0

def increment_usage(scope, feature, date=None):
    """Increment usage count for a specific feature"""
    if date is None:
        date = get_today_date()
    
    db = get_db()
    if not db.client:
        return False
    
    try:
        # Use upsert to create document if it doesn't exist or increment if it does
        update_field = f"counters.{feature}"
        db.usage.update_one(
            {'scope': scope, 'date': date},
            {'$inc': {update_field: 1}},
            upsert=True
        )
        return True
    except Exception as e:
        print(f"Error incrementing usage: {e}")
        return False

def check_limit(feature):
    """Check if current user/guest is within limits for a feature"""
    scope = get_current_scope()
    user_type = get_user_type()
    
    current_count = get_usage_count(scope, feature)
    limit = LIMITS[user_type][feature]
    
    return {
        'allowed': current_count < limit,
        'current': current_count,
        'limit': limit,
        'user_type': user_type,
        'scope': scope
    }

def track_usage(feature):
    """Track usage for a feature and return success/failure"""
    scope = get_current_scope()
    return increment_usage(scope, feature)

def get_active_share_links_count(user_id):
    """Get count of active share links for a user"""
    db = get_db()
    if not db.client:
        return 0
    
    try:
        from bson import ObjectId
        count = db.share_links.count_documents({
            'user_id': ObjectId(user_id),
            'is_active': True,
            'expires_at': {'$gt': datetime.now(timezone.utc)}
        })
        return count
    except Exception as e:
        print(f"Error getting active share links count: {e}")
        return 0

def get_usage_summary(scope=None, date=None):
    """Get usage summary for debugging/admin purposes"""
    if scope is None:
        scope = get_current_scope()
    if date is None:
        date = get_today_date()
    
    db = get_db()
    if not db.client:
        return {}
    
    try:
        usage_doc = db.usage.find_one({'scope': scope, 'date': date})
        if not usage_doc or 'counters' not in usage_doc:
            return {'analyses': 0, 'ai_search': 0, 'share_links_created': 0}
        
        counters = usage_doc['counters']
        return {
            'analyses': counters.get('analyses', 0),
            'ai_search': counters.get('ai_search', 0),
            'share_links_created': counters.get('share_links_created', 0)
        }
    except Exception as e:
        print(f"Error getting usage summary: {e}")
        return {}


def _guest_v3_trials_collection():
    db = get_db()
    if getattr(db, "client", None) is None:
        return None
    return getattr(db, "guest_v3_trials", None)


def guest_v3_trial_status(guest_uuid):
    """Return {limit, used, remaining} for lifetime V3 guest meal trial."""
    coll = _guest_v3_trials_collection()
    if coll is None or not guest_uuid:
        return {"limit": GUEST_V3_TRIAL_LIMIT, "used": 0, "remaining": GUEST_V3_TRIAL_LIMIT}
    doc = coll.find_one({"_id": guest_uuid})
    used = int(doc.get("count", 0)) if doc else 0
    used = max(0, min(used, GUEST_V3_TRIAL_LIMIT))
    rem = max(0, GUEST_V3_TRIAL_LIMIT - used)
    return {"limit": GUEST_V3_TRIAL_LIMIT, "used": used, "remaining": rem}


def try_reserve_guest_v3_trial(guest_uuid):
    """Atomically consume one trial slot before expensive work (AI / IO). Returns True if reserved."""
    from pymongo import ReturnDocument

    coll = _guest_v3_trials_collection()
    if coll is None or not guest_uuid:
        return False
    coll.update_one({"_id": guest_uuid}, {"$setOnInsert": {"count": 0}}, upsert=True)
    after = coll.find_one_and_update(
        {"_id": guest_uuid, "count": {"$lt": GUEST_V3_TRIAL_LIMIT}},
        {"$inc": {"count": 1}},
        return_document=ReturnDocument.AFTER,
    )
    return after is not None


def refund_guest_v3_trial(guest_uuid):
    """Release one reserved slot if downstream work failed after try_reserve_guest_v3_trial."""
    coll = _guest_v3_trials_collection()
    if coll is None or not guest_uuid:
        return
    try:
        coll.update_one({"_id": guest_uuid, "count": {"$gt": 0}}, {"$inc": {"count": -1}})
    except Exception as e:
        print(f"Error refunding guest V3 trial: {e}")
