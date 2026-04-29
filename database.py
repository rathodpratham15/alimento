# database.py - MongoDB Atlas Database Manager - VERCEL FIXED VERSION
import os
import certifi
from pymongo import MongoClient
from datetime import datetime
import json
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING

# Load environment variables
load_dotenv()

class MongoDBManager:
    """Singleton-style wrapper around a PyMongo ``MongoClient``.

    Manages the connection lifecycle, collection references, and index
    creation for all Alimento collections in MongoDB Atlas.  When
    ``MONGODB_URI`` is absent the instance degrades gracefully: ``client``
    and ``db`` are ``None`` and all data-access methods return safe fallback
    values instead of raising.

    Collections exposed as attributes (each is a ``pymongo.Collection``):

    * ``collection`` – legacy ``analysis_history``
    * ``users``, ``logins``, ``usage``, ``share_links``, ``hydration_logs``
    * ``meal_logs``, ``food_items``, ``inventory_items``, ``inventory_meal_suggestions``, ``barcode_cache``
    * ``recipes``, ``meal_plans``, ``grocery_lists``, ``weight_logs``
    * ``chat_sessions``, ``challenges``, ``challenge_members``
    * ``activity_integrations``, ``notification_settings``, ``migration_state``
    * ``user_profiles``, ``nutrition_goals``, ``diet_preferences``

    Environment variables:

    * ``MONGODB_URI``        – MongoDB Atlas connection string (required).
    * ``AUTO_CREATE_INDEXES`` – Set to ``'1'`` to run :meth:`ensure_indexes`
      at startup; omit for faster cold starts (indexes created on demand).
    """

    def __init__(self):
        # Get connection string from environment variable
        self.connection_string = os.getenv('MONGODB_URI')
        print(f"MongoDB URI exists: {bool(self.connection_string)}")
        
        if not self.connection_string:
            print("MONGODB_URI not found in environment variables!")
            self.client = None
            self.db = None
            return
        
        # Print partial URI for debugging (hide password)
        uri_parts = self.connection_string.split('@')
        if len(uri_parts) > 1:
            print(f"Connecting to: ...@{uri_parts[1][:50]}...")
        
        try:
            # Use certifi CA bundle so TLS to Atlas works on macOS/Python builds
            # that lack a full system certificate store (avoids SSL verify failures
            # during auth and other first DB operations).
            self.client = MongoClient(
                self.connection_string,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=10000,  # 10 seconds timeout
                connectTimeoutMS=10000,
                socketTimeoutMS=10000,
                maxPoolSize=10,
                retryWrites=True
            )
            
            # Set database and collection
            self.db = self.client.diet_designer
            self.collection = self.db.analysis_history
            # Additional collections
            self.users = self.db.users
            self.logins = self.db.logins
            self.usage = self.db.usage  # Usage tracking
            self.share_links = self.db.share_links  # Shareable analysis links
            self.hydration_logs = self.db.hydration_logs  # Water intake per user/day

            # V3 feature collections
            self.meal_logs = self.db.meal_logs
            self.guest_v3_trials = self.db.guest_v3_trials
            self.food_items = self.db.food_items
            self.inventory_items = self.db.inventory_items
            self.inventory_meal_suggestions = self.db.inventory_meal_suggestions
            self.barcode_cache = self.db.barcode_cache
            self.recipes = self.db.recipes
            self.meal_plans = self.db.meal_plans
            self.grocery_lists = self.db.grocery_lists
            self.weight_logs = self.db.weight_logs
            self.chat_sessions = self.db.chat_sessions
            self.challenges = self.db.challenges
            self.challenge_members = self.db.challenge_members
            self.activity_integrations = self.db.activity_integrations
            self.notification_settings = self.db.notification_settings
            self.migration_state = self.db.migration_state
            
            # User Profile collections
            self.user_profiles = self.db.user_profiles
            self.nutrition_goals = self.db.nutrition_goals
            self.diet_preferences = self.db.diet_preferences

            self._indexes_ensured = False
            if os.getenv('AUTO_CREATE_INDEXES', '0') == '1':
                self.ensure_indexes()
            else:
                print("MongoDB connected. Skipping index creation at startup.")
            
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            print(f"Error type: {type(e).__name__}")
            
            # More detailed error info
            if "authentication failed" in str(e).lower():
                print("Authentication issue - check username/password in MongoDB URI")
            elif "timeout" in str(e).lower():
                print("Connection timeout - check network/firewall settings")
            elif "dns" in str(e).lower():
                print("DNS resolution issue - check MongoDB cluster hostname")
            
            self.client = None
            self.db = None

    def ensure_indexes(self):
        """Create all required MongoDB indexes (idempotent).

        Safe to call multiple times; a flag prevents redundant operations in
        the same process lifetime.

        Returns:
            ``True`` on success, ``False`` if the database is not connected or
            index creation fails.
        """
        if not self.client:
            print("Cannot create indexes: database not connected")
            return False

        if self._indexes_ensured:
            return True

        try:
            # User indexes
            self.users.create_index([('email', ASCENDING)], unique=True, sparse=True)
            self.users.create_index([('google_sub', ASCENDING)], unique=True, sparse=True)

            # Analysis indexes
            self.collection.create_index([('created_at', ASCENDING)])
            self.collection.create_index([('user_id', ASCENDING)])
            self.collection.create_index([('guest_session_id', ASCENDING)])

            # Login tracking indexes
            self.logins.create_index([('when', ASCENDING)])

            # Usage tracking indexes (compound index for scope + date)
            self.usage.create_index([('scope', ASCENDING), ('date', ASCENDING)], unique=True)

            # Share links indexes
            self.share_links.create_index([('token', ASCENDING)], unique=True)
            self.share_links.create_index([('user_id', ASCENDING), ('is_active', ASCENDING)])
            self.share_links.create_index([('expires_at', ASCENDING)])

            # User Profile indexes
            self.user_profiles.create_index([('user_id', ASCENDING)], unique=True)
            self.nutrition_goals.create_index([('user_id', ASCENDING)])
            self.diet_preferences.create_index([('user_id', ASCENDING)], unique=True)

            # Hydration indexes
            self.hydration_logs.create_index([('user_id', ASCENDING), ('date', ASCENDING)], unique=True)

            # V3 indexes
            self.meal_logs.create_index([('user_id', ASCENDING), ('logged_at', DESCENDING)])
            self.meal_logs.create_index([('guest_session_id', ASCENDING), ('logged_at', DESCENDING)])
            self.meal_logs.create_index([('source', ASCENDING), ('created_at', DESCENDING)])
            self.meal_logs.create_index([('schema_version', ASCENDING)])

            self.food_items.create_index([('user_id', ASCENDING), ('created_at', DESCENDING)])
            self.inventory_items.create_index([('user_id', ASCENDING), ('location', ASCENDING)])
            self.inventory_items.create_index([('user_id', ASCENDING), ('name', ASCENDING)])
            self.inventory_items.create_index([('user_id', ASCENDING), ('expires_at', ASCENDING)])
            self.inventory_items.create_index([('user_id', ASCENDING), ('updated_at', DESCENDING)])

            self.inventory_meal_suggestions.create_index([('user_id', ASCENDING)], unique=True)

            self.barcode_cache.create_index([('barcode', ASCENDING)], unique=True)
            self.barcode_cache.create_index([('updated_at', DESCENDING)])

            self.recipes.create_index([('user_id', ASCENDING), ('created_at', DESCENDING)])
            self.recipes.create_index([('is_public', ASCENDING), ('diet_tags', ASCENDING)])

            self.meal_plans.create_index([('user_id', ASCENDING), ('week_start', ASCENDING)], unique=True)
            self.grocery_lists.create_index([('user_id', ASCENDING), ('week_start', ASCENDING)], unique=True)

            self.weight_logs.create_index([('user_id', ASCENDING), ('date', ASCENDING)], unique=True)
            self.weight_logs.create_index([('user_id', ASCENDING), ('created_at', DESCENDING)])

            self.chat_sessions.create_index([('user_id', ASCENDING), ('updated_at', DESCENDING)])

            self.challenges.create_index([('is_active', ASCENDING), ('created_at', DESCENDING)])
            self.challenges.create_index([('created_by', ASCENDING), ('created_at', DESCENDING)])

            self.challenge_members.create_index([('challenge_id', ASCENDING), ('user_id', ASCENDING)], unique=True)
            self.challenge_members.create_index([('user_id', ASCENDING), ('joined_at', DESCENDING)])

            self.activity_integrations.create_index([('user_id', ASCENDING), ('provider', ASCENDING)], unique=True)
            self.notification_settings.create_index([('user_id', ASCENDING)], unique=True)

            self.migration_state.create_index([('name', ASCENDING)], unique=True)

            self._indexes_ensured = True
            print("MongoDB indexes ensured successfully")
            return True
        except Exception as e:
            print(f"Index creation warning: {e}")
            return False
    
    def is_connected(self):
        """Ping MongoDB to verify the connection is alive.

        Returns:
            ``True`` if the ping succeeds, ``False`` otherwise (including
            when ``client`` is ``None``).
        """
        if not self.client:
            return False
        try:
            self.client.admin.command('ping')
            return True
        except:
            return False
    
    def save_analysis(self, analysis_data):
        """Persist a meal analysis document to the ``analysis_history`` collection.

        Automatically sets ``timestamp`` (ISO string) and ``created_at``
        (datetime) if absent.  Ownership fields ``user_id`` and
        ``guest_session_id`` default to ``None`` if not supplied.

        Args:
            analysis_data: Dict containing the analysis payload.  Modified
                           in-place to add metadata fields before insertion.

        Returns:
            ``{"success": True, "id": "<ObjectId str>"}`` on success, or
            ``{"success": False, "error": "<message>"}`` on failure.
        """
        if not self.client:
            return {"success": False, "error": "Database not connected"}
        
        try:
            # Add timestamp if not present (UTC)
            if 'timestamp' not in analysis_data:
                analysis_data['timestamp'] = datetime.utcnow().isoformat()
            
            # Add created_at for sorting (UTC)
            analysis_data['created_at'] = datetime.utcnow()

            # Ensure ownership fields exist (nullable)
            analysis_data.setdefault('user_id', None)
            analysis_data.setdefault('guest_session_id', None)
            
            print(f"Attempting to save analysis to MongoDB...")
            
            # Insert document
            result = self.collection.insert_one(analysis_data)
            
            print(f"Analysis saved with ID: {result.inserted_id}")
            return {"success": True, "id": str(result.inserted_id)}
            
        except Exception as e:
            print(f"Save error: {e}")
            print(f"Error type: {type(e).__name__}")
            return {"success": False, "error": str(e)}
    
    def get_history(self, limit=20):
        """Retrieve recent meal analyses sorted newest-first.

        Args:
            limit: Maximum number of documents to return (default ``20``).

        Returns:
            List of analysis dicts with ``_id`` converted to string and a
            ``timestamp`` ISO string derived from ``created_at``.  Returns
            ``[]`` if the database is not connected or on error.
        """
        if not self.client:
            print("Database not connected, returning empty history")
            return []
        
        try:
            print(f"Attempting to retrieve {limit} analyses...")
            
            # Get documents sorted by created_at (newest first)
            cursor = self.collection.find().sort("created_at", -1).limit(limit)
            
            # Convert to list and handle ObjectId serialization
            history = []
            for doc in cursor:
                # Convert ObjectId to string for JSON serialization
                doc['_id'] = str(doc['_id'])
                
                # Ensure timestamp is in the format expected by templates
                if 'created_at' in doc:
                    doc['timestamp'] = doc['created_at'].isoformat()
                
                history.append(doc)
            
            print(f"Retrieved {len(history)} analyses from database")
            return history
            
        except Exception as e:
            print(f"History retrieval error: {e}")
            print(f"Error type: {type(e).__name__}")
            return []
    
    def delete_analysis(self, analysis_id):
        """Delete a single analysis document by its string ObjectId.

        Args:
            analysis_id: Hexadecimal ObjectId string of the document to delete.

        Returns:
            ``{"success": True, "message": "..."}`` if deleted,
            ``{"success": False, "error": "..."}`` if not found or on error.
        """
        if not self.client:
            return {"success": False, "error": "Database not connected"}
        
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(analysis_id)
            
            # Delete document
            result = self.collection.delete_one({"_id": object_id})
            
            if result.deleted_count > 0:
                print(f"Deleted analysis with ID: {analysis_id}")
                return {"success": True, "message": "Analysis deleted successfully"}
            else:
                return {"success": False, "error": "Analysis not found"}
                
        except Exception as e:
            print(f"Delete error: {e}")
            return {"success": False, "error": str(e)}
    
    def clear_all_history(self):
        """Delete every document in the ``analysis_history`` collection.

        Returns:
            ``{"success": True, "message": "Cleared N analyses"}`` on success,
            or ``{"success": False, "error": "..."}`` on failure.
        """
        if not self.client:
            return {"success": False, "error": "Database not connected"}
        
        try:
            # Delete all documents
            result = self.collection.delete_many({})
            
            print(f"Cleared {result.deleted_count} analyses from database")
            return {"success": True, "message": f"Cleared {result.deleted_count} analyses"}
            
        except Exception as e:
            print(f"Clear history error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_stats(self):
        """Return basic database and collection statistics.

        Returns:
            Dict containing at minimum ``total_analyses`` (int) and
            ``connected`` (bool).  When data exists, also includes
            ``database_size`` and ``collection_size`` in bytes if the
            server supports ``dbStats`` / ``collStats`` commands.
            Returns ``{"error": "..."}`` when the database is not connected.
        """
        if not self.client:
            return {"error": "Database not connected"}
        
        try:
            total_analyses = self.collection.count_documents({})
            
            stats = {
                "total_analyses": total_analyses,
                "connected": True
            }
            
            # Only get advanced stats if we have data
            if total_analyses > 0:
                try:
                    db_stats = self.db.command("dbStats")
                    stats["database_size"] = db_stats.get("dataSize", 0)
                    
                    coll_stats = self.db.command("collStats", "analysis_history")
                    stats["collection_size"] = coll_stats.get("size", 0)
                except:
                    # Advanced stats failed, but basic stats work
                    pass
            
            return stats
            
        except Exception as e:
            print(f"Stats error: {e}")
            return {"error": str(e), "connected": False}

# Global database instance
db_manager = None

def get_db():
    """Return the global :class:`MongoDBManager` singleton, creating it lazily.

    Subsequent calls return the same instance without reconnecting.  Safe to
    call inside Flask's ``LocalProxy`` context.

    Returns:
        The module-level :class:`MongoDBManager` instance.
    """
    global db_manager
    if db_manager is None:
        print("Initializing MongoDB connection...")
        db_manager = MongoDBManager()
    return db_manager
