from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
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
import base64
import requests

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# API Keys  
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/kfi0ulqzkpuu5e"
RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY') or os.environ.get('REPLICATE_API_TOKEN')  # Try RunPod key first, fallback to existing
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_ASSISTANT_ID = "asst_dosMuAyLnY9vqMvVnAtO5GHx"  # Your Interior Design Assistant

# Create the main app without a prefix
app = FastAPI(title="ProAgentTools", description="AI-powered tools for real estate agents")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Ensure uploads directory exists
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

# Interior Design Configuration Data
ROOM_TYPES = [
    {"id": "living_room", "name": "Living Room"},
    {"id": "bedroom", "name": "Bedroom"},
    {"id": "kitchen", "name": "Kitchen"},
    {"id": "dining_room", "name": "Dining Room"},
    {"id": "bathroom", "name": "Bathroom"}
]

DESIGNERS = [
    {"id": "alessia_duval", "name": "Alessia Duval", "description": "Infuses Parisian elegance with global eclecticism, layering vibrant textiles and refined antiques."},
    {"id": "adrian_mercer", "name": "Adrian Mercer", "description": "Transforms post-industrial materials into poetic, sculptural interiors full of moody drama."},
    {"id": "lucien_hart", "name": "Lucien Hart", "description": "Fuses runway glamour with architectural audacity, delivering bold, theatrical spaces saturated in jewel tones."},
    {"id": "elinor_hartwell", "name": "Elinor Hartwell", "description": "Creates warm, soulful rooms where mindful living meets tactile, handcrafted comfort."},
    {"id": "bianca_morelli", "name": "Bianca Morelli", "description": "Weaves fluid, organic forms and tactile layers into emotionally engaging, elegant interiors."},
    {"id": "eleanor_reed", "name": "Eleanor Reed", "description": "Mixes vintage patina with contemporary comfort for richly textured, eclectic authenticity."},
    {"id": "oliver_renard", "name": "Oliver Renard", "description": "Stages maximalist fantasies with jewel-tone palettes, luxe textures, and theatrical storytelling."},
    {"id": "gabrielle_marlowe", "name": "Gabrielle Marlowe", "description": "Blends Southern graciousness with European classicism to craft airy, refined spaces of quiet luxury."},
    {"id": "elise_marceau", "name": "Elise Marceau", "description": "Balances minimalist restraint with tactile warmth, creating zen-like sanctuaries of European elegance."},
    {"id": "alexander_bennett", "name": "Alexander Bennett", "description": "Revives classical grandeur with tailored American sophistication and rich architectural detailing."},
    {"id": "allegra_marquez", "name": "Allegra Marquez", "description": "Combines cultural authenticity with modern lines, marrying vibrant heritage motifs to Scandinavian restraint."},
    {"id": "olivia_bennett", "name": "Olivia Bennett", "description": "Creates approachable elegance through thoughtful styling and sustainable, handcrafted details."}
]

COLOR_SCHEMES = [
    {"id": "glacial_muse", "name": "Glacial Muse", "description": "A tranquil blend of icy pastels and frosted neutrals evoking Nordic serenity and snow-dappled calm."},
    {"id": "nomad_prism", "name": "Nomad Prism", "description": "A kaleidoscope of vibrant gems and wanderlust tones, perfect for free spirits with eclectic tastes."},
    {"id": "urban_alloy", "name": "Urban Alloy", "description": "A cold fusion of iron hues and industrial patina—gritty, architectural, and unapologetically raw."},
    {"id": "aegean_whisper", "name": "Aegean Whisper", "description": "Salty breeze in a palette—oceanic blues and sun-kissed earth tones conjure Mediterranean leisure."},
    {"id": "velvet_deco", "name": "Velvet Deco", "description": "Moody, moony, and maximalist. Deep jewel tones, metallic flourishes, and cinematic glamour."},
    {"id": "desert_modern", "name": "Desert Modern", "description": "Warmed by sand and cactus shadow, this scheme blends burnt earth and washed neutrals with desert grace."},
    {"id": "enchanted_forest", "name": "Enchanted Forest", "description": "Lush emeralds and bark browns meet mossy whispers—an ode to deep woods and fairy tale glades."},
    {"id": "savannah_bloom", "name": "Savannah Bloom", "description": "Sunburnt petals and golden grass—this palette hums with the wild, untamed joy of African summers."},
    {"id": "canyon_clay", "name": "Canyon Clay", "description": "Terracotta cliffs under a molten sky. Rust, clay, and sun-scorched neutrals layer like canyon walls."},
    {"id": "lunar_drift", "name": "Lunar Drift", "description": "Muted moonlight and futuristic haze—icy greys, pale lavenders, and shadows in motion."},
    {"id": "sienna_smoke", "name": "Sienna Smoke", "description": "Warm neutrals drift through dusty plumes of clay and chalk—effortlessly grounded and elegant."},
    {"id": "retro_zest", "name": "Retro Zest", "description": "A punchy throwback of avocado green, popsicle orange, and lemony optimism—your cool aunt's kitchen."},
    {"id": "twilight_grove", "name": "Twilight Grove", "description": "Evening rain on bark and bloom. Smoky violet, ash green, and the hush of forest shadows."},
    {"id": "citrus_pop", "name": "Citrus Pop", "description": "Grapefruit zest and neon fizz—this is breakfast at sunrise with sunglasses on."},
    {"id": "oxblood_study", "name": "Oxblood Study", "description": "Oxblood, ink, and old paper tones—academic without the arrogance, rich with ritual."},
    {"id": "sunken_studio", "name": "Sunken Studio", "description": "An undersea study in moody ink, shale, and studio-light neutrals. Moody, tactile, introspective."},
    {"id": "charred_cotton", "name": "Charred Cotton", "description": "Ash, linen, and charcoal smudge together like erased sketches on vintage paper."},
    {"id": "silken_ember", "name": "Silken Ember", "description": "Firelight meets silk scarf—subdued luxury with a whisper of spice and after-hours warmth."},
    {"id": "mineral_tonic", "name": "Mineral Tonic", "description": "This tonic blends mineral blue, flint, and dried herbs for a grounded yet experimental harmony."},
    {"id": "bauhaus_dusk", "name": "Bauhaus Dusk", "description": "A modernist poem in color—primary accents on a bed of greys and grounded pastels."}
]

# Models
class InteriorDesignRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_filename: str
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "processing"
    processed_image_url: Optional[str] = None
    error_message: Optional[str] = None
    prediction_id: Optional[str] = None
    room_type: Optional[str] = None
    designer: Optional[str] = None
    color_scheme: Optional[str] = None
    generated_prompt: Optional[str] = None

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

# Interior Design Configuration Endpoints
@api_router.get("/interior-design/room-types")
async def get_room_types():
    """Get available room types"""
    return {"room_types": ROOM_TYPES}

@api_router.get("/interior-design/designers") 
async def get_designers():
    """Get available interior designers"""
    return {"designers": DESIGNERS}

@api_router.get("/interior-design/color-schemes")
async def get_color_schemes():
    """Get available color schemes"""
    return {"color_schemes": COLOR_SCHEMES}

# Interior Design Model Routes
async def generate_design_prompt_with_assistant(room_type: str, designer: str, color_scheme: str) -> str:
    """Generate design prompt using OpenAI Assistant"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Create a thread
        thread = client.beta.threads.create()
        
        # Create message with user preferences
        message_content = f"""
        Please generate a detailed interior design prompt for:
        
        Room Type: {room_type}
        Designer Style: {designer}
        Color Scheme: {color_scheme}
        
        The prompt should be optimized for AI image generation and include specific details about furniture, lighting, textures, and overall aesthetic.
        """
        
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message_content
        )
        
        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=OPENAI_ASSISTANT_ID
        )
        
        # Wait for completion
        import time
        max_wait = 30  # 30 seconds max
        wait_time = 0
        while run.status in ["queued", "in_progress"] and wait_time < max_wait:
            time.sleep(2)
            wait_time += 2
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        
        if run.status == "completed":
            # Get messages
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            assistant_message = messages.data[0].content[0].text.value
            return assistant_message
        else:
            logger.error(f"Assistant run failed with status: {run.status}")
            return f"A {designer} style {room_type} with {color_scheme} color scheme, featuring modern furniture and elegant lighting"
            
    except Exception as e:
        logger.error(f"OpenAI Assistant error: {str(e)}")
        # Fallback prompt
        return f"A {designer} style {room_type} with {color_scheme} color scheme, featuring modern furniture and elegant lighting"

@api_router.post("/interior-design/process")
async def process_interior_design(
    file: UploadFile = File(...),
    room_type: str = "living_room",
    designer: str = "alessia_duval", 
    color_scheme: str = "glacial_muse"
):
    """Process an interior image with custom design preferences"""
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
        
        # Generate custom prompt using OpenAI Assistant
        logger.info(f"Generating prompt for room_type={room_type}, designer={designer}, color_scheme={color_scheme}")
        generated_prompt = await generate_design_prompt_with_assistant(room_type, designer, color_scheme)
        logger.info(f"Generated prompt: {generated_prompt[:100]}...")
        
        # Create database record with preferences
        design_request = InteriorDesignRequest(
            original_filename=file.filename,
            status="processing",
            room_type=room_type,
            designer=designer,
            color_scheme=color_scheme,
            generated_prompt=generated_prompt
        )
        
        await db.interior_designs.insert_one(design_request.dict())
        
        # TEMPORARY: Switch to working Replicate model while RunPod is being fixed
        try:
            import replicate
            
            with open(temp_file_path, "rb") as image_file:
                # Use a working interior design model with custom prompt
                output = replicate.run(
                    "adirik/interior-design:76604baddc85b1b4616e1c6475eca080da339c8875bd4996705440484a6eac38",
                    input={
                        "image": image_file,
                        "prompt": generated_prompt  # Use the AI-generated prompt
                    }
                )
                
                # Extract URL from output
                if hasattr(output, 'url'):
                    processed_url = str(output.url)
                elif isinstance(output, str):
                    processed_url = output
                elif isinstance(output, list) and len(output) > 0:
                    if hasattr(output[0], 'url'):
                        processed_url = str(output[0].url)
                    else:
                        processed_url = str(output[0])
                else:
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
                    "original_filename": file.filename,
                    "room_type": room_type,
                    "designer": designer,
                    "color_scheme": color_scheme,
                    "generated_prompt": generated_prompt,
                    "message": "Image processed with custom design preferences"
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
                
            raise HTTPException(status_code=500, detail=f"Failed to start processing: {error_msg}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@api_router.get("/interior-design/status/{design_id}")
async def check_design_status(design_id: str):
    """Check the status of an interior design processing job"""
    try:
        design = await db.interior_designs.find_one({"id": design_id})
        if not design:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # If we have a prediction ID and status is not completed/failed, check RunPod
        if design.get("prediction_id") and design.get("status") in ["submitted", "processing"]:
            try:
                # Check status with RunPod API
                headers = {
                    "Authorization": f"Bearer {RUNPOD_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                status_response = requests.get(
                    f"{RUNPOD_ENDPOINT}/status/{design['prediction_id']}",
                    headers=headers
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    job_status = status_data.get("status")
                    
                    if job_status == "COMPLETED":
                        # Extract the output image URL
                        output_data = status_data.get("output")
                        processed_url = None
                        
                        if isinstance(output_data, dict):
                            processed_url = output_data.get("image_url") or output_data.get("output")
                        elif isinstance(output_data, list) and len(output_data) > 0:
                            processed_url = output_data[0]
                        elif isinstance(output_data, str):
                            processed_url = output_data
                        
                        if processed_url:
                            # Update database with success
                            await db.interior_designs.update_one(
                                {"id": design_id},
                                {"$set": {
                                    "status": "completed",
                                    "processed_image_url": processed_url
                                }}
                            )
                            
                            return {
                                "id": design_id,
                                "status": "completed",
                                "processed_image_url": processed_url,
                                "original_filename": design.get("original_filename")
                            }
                        
                    elif job_status == "FAILED":
                        error_msg = status_data.get("error", "Unknown error occurred")
                        await db.interior_designs.update_one(
                            {"id": design_id},
                            {"$set": {
                                "status": "failed",
                                "error_message": error_msg
                            }}
                        )
                        
                        return {
                            "id": design_id,
                            "status": "failed",
                            "error_message": error_msg
                        }
                        
                    elif job_status in ["IN_QUEUE", "IN_PROGRESS"]:
                        # Update status but don't change anything else
                        await db.interior_designs.update_one(
                            {"id": design_id},
                            {"$set": {"status": "processing"}}
                        )
                        
                        return {
                            "id": design_id,
                            "status": "processing",
                            "message": "Your image is being processed by RunPod. This may take 2-3 minutes."
                        }
                    
                    elif job_status == "IN_QUEUE":
                        # Check if job has been stuck in queue too long
                        design_time = design.get("upload_timestamp")
                        if design_time:
                            from datetime import datetime, timedelta
                            time_diff = datetime.utcnow() - design_time
                            if time_diff > timedelta(minutes=5):  # Stuck for more than 5 minutes
                                await db.interior_designs.update_one(
                                    {"id": design_id},
                                    {"$set": {
                                        "status": "failed",
                                        "error_message": "RunPod workers are currently experiencing issues. Please try again later."
                                    }}
                                )
                                
                                return {
                                    "id": design_id,
                                    "status": "failed",
                                    "error_message": "RunPod workers are currently experiencing issues. Please try again later."
                                }
                        
                        return {
                            "id": design_id,
                            "status": "processing", 
                            "message": "Your image is queued for processing. RunPod workers are starting up..."
                        }
                
            except Exception as e:
                logger.error(f"Error checking RunPod status: {str(e)}")
                # Return current database status if RunPod check fails
        
        # Return current database status
        return {
            "id": design_id,
            "status": design.get("status", "unknown"),
            "processed_image_url": design.get("processed_image_url"),
            "error_message": design.get("error_message"),
            "original_filename": design.get("original_filename")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check status: {str(e)}")

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