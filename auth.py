from flask import Blueprint, redirect, url_for, request, current_app, jsonify, flash, make_response, render_template, session
from authlib.integrations.flask_client import OAuth
from flask_login import UserMixin
from dotenv import load_dotenv
from flask_login import login_user, logout_user, current_user
from itsdangerous import URLSafeSerializer
from datetime import datetime
import os
import uuid
from werkzeug.local import LocalProxy
from database import get_db
from bson import ObjectId

load_dotenv()

auth_bp = Blueprint('auth', __name__)

db = LocalProxy(get_db)

# OAuth setup - will be initialized when blueprint is registered
oauth = OAuth()

GUEST_COOKIE_NAME = 'guest_session'


def _guest_cookie_serializer():
    """Match ``app.secret_key`` so guest cookies from ``after_request`` verify here."""
    return URLSafeSerializer(current_app.secret_key, salt='guest-session')


def init_oauth(app):
    """Initialize OAuth with Flask app"""
    oauth.init_app(app)
    oauth.register(
        name='google',
        client_id=os.getenv('GOOGLE_CLIENT_ID'),
        client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )


@auth_bp.route('/login')
def login():
    # Build a redirect URI that matches this server's external URL to avoid mismatch
    try:
        redirect_uri = url_for('auth.auth_callback', _external=True)
    except Exception:
        redirect_uri = os.getenv('OAUTH_REDIRECT_URI', 'http://localhost:5000/auth/callback')

    # If this endpoint is called via browser navigation, show a page; otherwise, start redirect
    # Preserve optional "next" param across the OAuth flow
    next_target = request.args.get('next')
    if next_target:
        session['next_after_login'] = next_target

    if request.args.get('ui') == '1':
        return render_template('login.html', oauth_redirect=redirect_uri)
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route('/register')
def register():
    # A simple register page that explains sign-in via Google and links to /login?ui=1
    return render_template('register.html')


@auth_bp.route('/auth/callback')
def auth_callback():
    try:
        token = oauth.google.authorize_access_token()
        # Get userinfo from Google's userinfo endpoint
        resp = oauth.google.get('https://www.googleapis.com/oauth2/v2/userinfo')
        userinfo = resp.json()

        # Upsert user - Google returns 'id' field, not 'sub'
        google_sub = userinfo.get('id') or userinfo.get('sub')  # Try both formats
        email = userinfo.get('email')
        name = userinfo.get('name')
        picture = userinfo.get('picture')
        
        if not google_sub:
            print(f"ERROR: google_sub is None! userinfo: {userinfo}")
            return redirect(url_for('index'))

        now = datetime.utcnow()
        # Prefer picture from userinfo, fallback handled earlier
        update = {
            'google_sub': google_sub,
            'email': email,
            'name': name,
            'picture': picture,
            'last_login_at': now
        }
        user_doc = db.users.find_one_and_update(
            {'google_sub': google_sub},
            {'$set': update, '$setOnInsert': {'created_at': now}},
            upsert=True, return_document=True
        )

        # Build user object for login (use the User class from app.py)
        user_id = str(user_doc.get('_id'))
        
        # Import User class from main app
        from app import User
        user = User(user_doc)
        login_user(user)

        # Insert login record
        try:
            login_record = {
                'user_id': ObjectId(user_id),
                'email': email,
                'when': now,
                'ip': request.remote_addr,
                'user_agent': request.headers.get('User-Agent')
            }
            db.logins.insert_one(login_record)
        except Exception:
            pass

        # Merge guest analyses if cookie exists
        guest_cookie = request.cookies.get(GUEST_COOKIE_NAME)
        response = make_response(redirect(url_for('index')))
        if guest_cookie:
            try:
                gid = _guest_cookie_serializer().loads(guest_cookie)
                # Move analyses from guest_session_id to user_id
                result = db.collection.update_many({'guest_session_id': gid}, {'$set': {'user_id': ObjectId(user_id), 'guest_session_id': None}})
                try:
                    db.meal_logs.update_many(
                        {'guest_session_id': gid},
                        {'$set': {'user_id': ObjectId(user_id), 'guest_session_id': None}},
                    )
                except Exception:
                    pass
                # Clear guest cookie
                response.set_cookie(GUEST_COOKIE_NAME, '', expires=0)
                flash('We moved your previous analyses to your account', 'success')
            except Exception:
                pass

        # Redirect to next if present and safe
        next_url = session.pop('next_after_login', None)
        if next_url and next_url.startswith('/'):
            return redirect(next_url)
        return redirect(url_for('history')) if not response.location else response
    except Exception as e:
        print(f"Auth callback error: {e}")
        return redirect(url_for('index'))


@auth_bp.route('/logout', methods=['POST'])
def logout():
    logout_user()
    response = make_response(redirect(url_for('index')))
    # After logout, issue a fresh guest cookie
    gid = str(uuid.uuid4())
    signed = _guest_cookie_serializer().dumps(gid)
    response.set_cookie(GUEST_COOKIE_NAME, signed, httponly=True, samesite='Lax', secure=bool(os.getenv('PRODUCTION')))
    return response


@auth_bp.route('/profile')
def profile():
    # Profile/settings page for logged-in users
    if not (current_user and getattr(current_user, 'is_authenticated', False)):
        return redirect(url_for('auth.login') + '?ui=1')

    try:
        user_id = current_user.get_id()
        user_doc = db.users.find_one({'_id': ObjectId(user_id)})
        
        # Get recent login history (last 10 logins)
        login_history = list(db.logins.find(
            {'user_id': ObjectId(user_id)}
        ).sort('when', -1).limit(10))
        
        # Format dates for display
        if user_doc:
            if 'created_at' in user_doc and user_doc['created_at']:
                user_doc['created_at_formatted'] = user_doc['created_at'].strftime('%B %d, %Y')
            else:
                user_doc['created_at_formatted'] = 'Unknown'
                
            if 'last_login_at' in user_doc and user_doc['last_login_at']:
                user_doc['last_login_at_formatted'] = user_doc['last_login_at'].strftime('%B %d, %Y at %I:%M %p')
                user_doc['last_login_iso'] = user_doc['last_login_at'].isoformat() + 'Z'
            else:
                user_doc['last_login_at_formatted'] = 'Unknown'
                user_doc['last_login_iso'] = ''
        
        # Format login history
        for login in login_history:
            login['when_formatted'] = login['when'].strftime('%B %d, %Y at %I:%M %p')
            login['when_iso'] = login['when'].isoformat() + 'Z'
            login['ip'] = login.get('ip', 'Unknown')
            login['user_agent_short'] = get_browser_info(login.get('user_agent', ''))
        
    except Exception as e:
        print(f"Profile error: {e}")
        user_doc = None
        login_history = []

    return render_template('profile.html', user=user_doc, login_history=login_history)

def get_browser_info(user_agent):
    """Extract browser and OS info from user agent string"""
    if not user_agent:
        return 'Unknown Browser'
    
    # Simple browser detection
    if 'Chrome' in user_agent:
        browser = 'Chrome'
    elif 'Firefox' in user_agent:
        browser = 'Firefox'
    elif 'Safari' in user_agent and 'Chrome' not in user_agent:
        browser = 'Safari'
    elif 'Edge' in user_agent:
        browser = 'Edge'
    else:
        browser = 'Unknown'
    
    # Simple OS detection
    if 'Windows' in user_agent:
        os_name = 'Windows'
    elif 'Macintosh' in user_agent or 'Mac OS' in user_agent:
        os_name = 'macOS'
    elif 'Linux' in user_agent:
        os_name = 'Linux'
    elif 'iPhone' in user_agent:
        os_name = 'iOS'
    elif 'Android' in user_agent:
        os_name = 'Android'
    else:
        os_name = 'Unknown'
    
    return f'{browser} on {os_name}'


@auth_bp.route('/api/me')
def api_me():
    if current_user and getattr(current_user, 'is_authenticated', False):
        return jsonify({'authenticated': True, 'user': {'id': current_user.id, 'email': getattr(current_user, 'email', None), 'name': getattr(current_user, 'name', None), 'picture': getattr(current_user, 'picture', None)}})
    # ensure guest cookie exists
    from usage_tracker import guest_v3_trial_status

    guest = request.cookies.get(GUEST_COOKIE_NAME)
    if not guest:
        gid = str(uuid.uuid4())
        signed = _guest_cookie_serializer().dumps(gid)
        response = make_response(jsonify({'authenticated': False, 'user': None, 'guest_mode': True, 'guest_trial': guest_v3_trial_status(gid)}))
        response.set_cookie(GUEST_COOKIE_NAME, signed, httponly=True, samesite='Lax', secure=bool(os.getenv('PRODUCTION')))
        return response
    try:
        gid = _guest_cookie_serializer().loads(guest)
        trial = guest_v3_trial_status(gid)
    except Exception:
        # Bad or legacy signature: replace cookie so v3 and after_request agree on a valid session.
        gid = str(uuid.uuid4())
        signed = _guest_cookie_serializer().dumps(gid)
        response = make_response(
            jsonify(
                {
                    'authenticated': False,
                    'user': None,
                    'guest_mode': True,
                    'guest_trial': guest_v3_trial_status(gid),
                }
            )
        )
        response.set_cookie(GUEST_COOKIE_NAME, signed, httponly=True, samesite='Lax', secure=bool(os.getenv('PRODUCTION')))
        return response
    return jsonify({'authenticated': False, 'user': None, 'guest_mode': True, 'guest_trial': trial})

