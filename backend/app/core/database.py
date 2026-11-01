import os
import logging
from app.core.mock_db import MockUser, MockCallLog

logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "")

client = None
db_connected = False

# Default to mock models immediately so the app is always functional
User = MockUser
CallLog = MockCallLog


async def connect_db():
    global client, db_connected, User, CallLog
    
    # Check if motor and beanie are available
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        from beanie import init_beanie
    except ImportError:
        logger.info("Motor or Beanie not installed. Running in mock database mode.")
        User = MockUser
        CallLog = MockCallLog
        db_connected = False
        return

    # In serverless environments (Vercel) without remote Atlas URI, default directly to mock DB
    if not MONGO_URI or ("localhost" in MONGO_URI and (os.getenv("VERCEL") or os.getenv("TESTING"))):
        logger.info("No remote MongoDB URI configured. Using fast mock database.")
        User = MockUser
        CallLog = MockCallLog
        db_connected = False
        return

    try:
        from app.models.user import User as RealUser
        from app.models.call_log import CallLog as RealCallLog
        
        # Use a short 2000ms timeout to avoid hanging serverless cold starts
        client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        await init_beanie(
            database=client.get_default_database(),
            document_models=[RealUser, RealCallLog],
        )
        User = RealUser
        CallLog = RealCallLog
        db_connected = True
        logger.info("Connected to MongoDB Atlas.")
    except Exception as e:
        db_connected = False
        logger.warning(f"Failed to connect to MongoDB ({e}). Running in MOCK MODE.")
        client = None
        User = MockUser
        CallLog = MockCallLog


async def disconnect_db():
    global client, db_connected
    if client:
        try:
            client.close()
            logger.info("Disconnected from MongoDB.")
        except Exception:
            pass
    db_connected = False


def get_client():
    return client


def is_db_connected() -> bool:
    return db_connected
