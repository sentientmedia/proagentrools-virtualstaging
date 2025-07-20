from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import replicate
import tempfile
import shutil
from emergentintegrations.llm.chat import LlmChat, UserMessage
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# API Keys
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Create the main app without a prefix
app = FastAPI(title="ProAgentTools", description="AI-powered tools for real estate agents")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Ensure uploads directory exists
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

# Models
class InteriorDesignRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_filename: str
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "processing"
    processed_image_url: Optional[str] = None
    error_message: Optional[str] = None

class GPTConceptRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    concept_type: str
    input_data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    response: Optional[str] = None

class GPTConceptResponse(BaseModel):
    id: str
    concept_type: str
    response: str
    timestamp: datetime

# Interior Design Model Routes
@api_router.post("/interior-design/process")
async def process_interior_design(file: UploadFile = File(...)):
    """Process an interior image using the trained Replicate model"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create unique filename and save temporarily
        file_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        temp_filename = f"{file_id}.{file_extension}"
        temp_file_path = uploads_dir / temp_filename
        
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create database record
        design_request = InteriorDesignRequest(
            original_filename=file.filename,
            status="processing"
        )
        
        await db.interior_designs.insert_one(design_request.dict())
        
        # Process with Replicate using your custom trained model
        try:
            with open(temp_file_path, "rb") as image_file:
                # Using your custom trained ProAgentTools interior design model
                output = replicate.run(
                    "deployments/sentientmedia/proagenttools25",
                    input={"image": image_file},
                    api_token=REPLICATE_API_TOKEN
                )
            
            # Extract URL from output - handle different Replicate response formats
            if hasattr(output, 'url'):
                # FileOutput object
                processed_url = str(output.url)
            elif isinstance(output, str):
                # Direct URL string
                processed_url = output
            elif isinstance(output, list) and len(output) > 0:
                # List of URLs or FileOutput objects
                if hasattr(output[0], 'url'):
                    processed_url = str(output[0].url)
                else:
                    processed_url = str(output[0])
            else:
                # Try to convert to string
                processed_url = str(output)
            
            # Update database with success
            await db.interior_designs.update_one(
                {"id": design_request.id},
                {"$set": {
                    "status": "completed",
                    "processed_image_url": processed_url
                }}
            )
            
            # Clean up temp file
            temp_file_path.unlink()
            
            return {
                "id": design_request.id,
                "status": "completed",
                "processed_image_url": processed_url,
                "original_filename": file.filename
            }
            
        except Exception as replicate_error:
            # Update database with error
            error_msg = str(replicate_error)
            await db.interior_designs.update_one(
                {"id": design_request.id},
                {"$set": {
                    "status": "failed",
                    "error_message": error_msg
                }}
            )
            
            # Clean up temp file
            if temp_file_path.exists():
                temp_file_path.unlink()
                
            raise HTTPException(status_code=500, detail=f"Interior design processing failed: {error_msg}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@api_router.get("/interior-design/history")
async def get_interior_design_history():
    """Get user's interior design processing history"""
    try:
        designs_list = []
        async for design in db.interior_designs.find().sort("upload_timestamp", -1).limit(20):
            # Convert ObjectId to string for JSON serialization
            if '_id' in design:
                design['_id'] = str(design['_id'])
            designs_list.append(design)
        return {"designs": designs_list}
    except Exception as e:
        logger.error(f"Error retrieving interior design history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

# GPT Concept Wrapper Routes
@api_router.post("/gpt-concepts/property-description", response_model=GPTConceptResponse)
async def generate_property_description(request: Dict[str, Any]):
    """Generate compelling property descriptions"""
    try:
        chat = LlmChat(
            api_key=OPENAI_API_KEY,
            session_id=str(uuid.uuid4()),
            system_message="You are a professional real estate copywriter. Create compelling, accurate property descriptions that highlight key features and appeal to potential buyers."
        ).with_model("openai", "gpt-4o")
        
        property_details = request.get("property_details", "")
        prompt = f"Create a professional property description for: {property_details}"
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        # Save to database
        concept_request = GPTConceptRequest(
            concept_type="property_description",
            input_data=request,
            response=response
        )
        await db.gpt_concepts.insert_one(concept_request.dict())
        
        return GPTConceptResponse(
            id=concept_request.id,
            concept_type="property_description",
            response=response,
            timestamp=concept_request.timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate description: {str(e)}")

@api_router.post("/gpt-concepts/market-analysis", response_model=GPTConceptResponse)
async def generate_market_analysis(request: Dict[str, Any]):
    """Generate market analysis for properties"""
    try:
        chat = LlmChat(
            api_key=OPENAI_API_KEY,
            session_id=str(uuid.uuid4()),
            system_message="You are a real estate market analyst. Provide detailed, data-driven market analysis and insights for properties and neighborhoods."
        ).with_model("openai", "gpt-4o")
        
        location = request.get("location", "")
        property_type = request.get("property_type", "")
        prompt = f"Provide a comprehensive market analysis for {property_type} properties in {location}. Include market trends, pricing insights, and investment potential."
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        # Save to database
        concept_request = GPTConceptRequest(
            concept_type="market_analysis",
            input_data=request,
            response=response
        )
        await db.gpt_concepts.insert_one(concept_request.dict())
        
        return GPTConceptResponse(
            id=concept_request.id,
            concept_type="market_analysis",
            response=response,
            timestamp=concept_request.timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate analysis: {str(e)}")

@api_router.post("/gpt-concepts/email-template", response_model=GPTConceptResponse)
async def generate_email_template(request: Dict[str, Any]):
    """Generate professional email templates for clients"""
    try:
        chat = LlmChat(
            api_key=OPENAI_API_KEY,
            session_id=str(uuid.uuid4()),
            system_message="You are a professional real estate communication specialist. Create polished, effective email templates that maintain professional tone while being personable and persuasive."
        ).with_model("openai", "gpt-4o")
        
        email_type = request.get("email_type", "")
        context = request.get("context", "")
        prompt = f"Create a professional {email_type} email template for real estate agents. Context: {context}"
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        # Save to database
        concept_request = GPTConceptRequest(
            concept_type="email_template",
            input_data=request,
            response=response
        )
        await db.gpt_concepts.insert_one(concept_request.dict())
        
        return GPTConceptResponse(
            id=concept_request.id,
            concept_type="email_template",
            response=response,
            timestamp=concept_request.timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate email: {str(e)}")

# Available GPT Concepts list
@api_router.get("/gpt-concepts/available")
async def get_available_concepts():
    """Get list of available GPT concepts"""
    concepts = [
        {
            "id": "property_description",
            "name": "Property Description Generator",
            "description": "Generate compelling property descriptions from basic details"
        },
        {
            "id": "market_analysis", 
            "name": "Market Analysis",
            "description": "Comprehensive market insights and trends analysis"
        },
        {
            "id": "email_template",
            "name": "Email Templates",
            "description": "Professional email templates for client communication"
        }
    ]
    return {"concepts": concepts}

# Get GPT Concepts History
@api_router.get("/gpt-concepts/history")
async def get_gpt_concepts_history():
    """Get user's GPT concepts usage history"""
    try:
        concepts_list = []
        async for concept in db.gpt_concepts.find().sort("timestamp", -1).limit(20):
            # Convert ObjectId to string for JSON serialization
            if '_id' in concept:
                concept['_id'] = str(concept['_id'])
            concepts_list.append(concept)
        return {"concepts": concepts_list}
    except Exception as e:
        logger.error(f"Error retrieving GPT concepts history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

# Main route
@api_router.get("/")
async def root():
    return {"message": "ProAgentTools API - AI-powered tools for real estate agents", "version": "1.0.0"}

# Health check
@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()