#!/usr/bin/env python3
"""
Script to create an admin user
Usage: python create_admin.py <email> <password> <full_name>
"""

import asyncio
import sys
import os
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from datetime import datetime
import uuid

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

# Setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def create_admin_user(email: str, password: str, full_name: str):
    """Create an admin user"""
    try:
        # Connect to database
        mongo_url = os.environ['MONGO_URL']
        client = AsyncIOMotorClient(mongo_url)
        db = client['proagenttools']
        
        # Check if admin already exists
        existing_admin = await db.admin_users.find_one({"email": email})
        if existing_admin:
            print(f"Admin user with email {email} already exists")
            return
        
        # Hash password
        hashed_password = pwd_context.hash(password)
        
        # Create admin user
        admin_user = {
            "id": str(uuid.uuid4()),
            "email": email,
            "full_name": full_name,
            "hashed_password": hashed_password,
            "role": "admin",
            "created_at": datetime.utcnow()
        }
        
        # Insert admin user
        await db.admin_users.insert_one(admin_user)
        print(f"Admin user created successfully: {email}")
        
    except Exception as e:
        print(f"Error creating admin user: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python create_admin.py <email> <password> <full_name>")
        sys.exit(1)
    
    email = sys.argv[1]
    password = sys.argv[2]
    full_name = sys.argv[3]
    
    asyncio.run(create_admin_user(email, password, full_name))