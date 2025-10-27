from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")

client = AsyncIOMotorClient(MONGODB_URL, server_api=ServerApi('1'))
db = client[DATABASE_NAME]

# Collections
teachers_collection = db.teachers
students_collection = db.students
sessions_collection = db.sessions
attendance_collection = db.attendance

async def connect_db():
    try:
        await client.admin.command('ping')
        print("✅ Connected to MongoDB!")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")

async def close_db():
    client.close()
    print("🔒 MongoDB connection closed")