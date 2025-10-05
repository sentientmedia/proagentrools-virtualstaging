from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import replicate
import tempfile
import shutil
from emergentintegrations.llm.chat import LlmChat, UserMessage
import asyncio
import base64
import requests
import aiohttp
import aiofiles
import jwt
import bcrypt
from passlib.context import CryptContext

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Create storage directories
PROCESSED_IMAGES_DIR = ROOT_DIR / "storage" / "processed_images"
PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Authentication
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Authentication helper functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def generate_referral_code() -> str:
    """Generate a unique referral code"""
    return str(uuid.uuid4()).replace('-', '')[:8].upper()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    user = await db.users.find_one({"id": user_id})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    return User(**user)

async def get_current_admin_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated admin user"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    admin = await db.admin_users.find_one({"id": user_id})
    if admin is None:
        raise HTTPException(status_code=401, detail="Admin access required")
    
    return AdminUser(**admin)

async def deduct_credits(user_id: str, credits_to_deduct: int) -> bool:
    """Deduct credits from user account"""
    user = await db.users.find_one({"id": user_id})
    if not user or user['credits'] < credits_to_deduct:
        return False
    
    await db.users.update_one(
        {"id": user_id},
        {"$inc": {"credits": -credits_to_deduct}}
    )
    return True

async def get_tool_rate(tool_name: str) -> int:
    """Get credit rate for a specific tool"""
    tool_config = await db.tool_rates.find_one({"tool_name": tool_name})
    if tool_config:
        return tool_config['credits_per_use']
    
    # Default rates if not configured
    default_rates = {
        'interior_design': 5,
        'gpt_concept': 1
    }
    return default_rates.get(tool_name, 1)

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
    {
        "id": "alessia_duval", 
        "name": "Alessia Duval", 
        "description": "Infuses Parisian elegance with global eclecticism, layering vibrant textiles and refined antiques.",
        "full_bio": "The Distinctive Designer: Alessia Duval\n\nEarly Life and Background\n\nBorn in the culturally diverse Marais district of Paris to a French antiques dealer mother and a father who was a Brazilian diplomat, Alessia Duval's earliest memories are colored with the international vibrancy of her upbringing. As a child, Alessia rarely lived in one place long; the diplomatic career of her father meant occasional relocations between Europe, South America, and North Africa. From each country, her mother encouraged her to collect handmade objects, exquisite textiles, and traditional crafts—small treasures carefully packed into aged trunks as they traveled onward. The foundational environment of her childhood cultivated Alessia's sophisticated yet eclectic palate, setting the cornerstone for a style which is passionately multicultural, explorative, yet precisely refined.\n\nEducation and Early Career\n\nAlessia pursued formal education in both fine arts and interior architecture, attending La Cambre National School of Visual Arts in Brussels, Belgium. Her tenure at La Cambre proved formative: under inspiring mentors in art and architecture, Alessia developed her rich visual vocabulary, blending historical elegance with contemporary lines. Post-graduation, she worked briefly in prestigious design studios in Antwerp and Milan, fine-tuning her intuitive understanding of luxury materials and artisanal detailing.\n\nAfter six years alongside noted European designers, Alessia deliberately sought experiences beyond continental Europe, embarking on extended trips to India, Japan, Morocco, and South America. With each stay, Alessia studied local artisanal techniques, collaborating closely with craftsmen to understand intricacies ranging from Moroccan zellige tiles to Japanese woodblock printing. These international experiences profoundly influenced her personal design philosophy, inspiring her distinctly layered, yet edited narrative approach to interiors.\n\nEvolution of Her Unique Style through Specific Influences\n\nAlessia's style evolved uniquely shaped by cross-cultural exchange. The intense textile colors of India left a lasting imprint, visible in Alessia's fearless combinations of hues. Japanese minimal simplicity taught her the critical skill of restraint, the power in thoughtful editing—each design element delivered purposefully, unobscured by superfluous ornamentation. North Africa influenced the visual flow and geometry of her spaces: intricate metallic work and architecture teaching her rhythm and depth. Her South American experiences inspired tactile warmth and rich storytelling through natural materials like the earthiness of Brazilian hardwoods or Andean wool textiles.\n\nMuch like Juan Montoya's curated precision, Alessia's style hinges on the ability to pair diverse elements while maintaining careful rigor of editing and spatial discipline. Vibrant eclecticism meets disciplined arrangement in each room she composes. Alessia describes her design philosophy as \"visual anthropology,\" aiming to narrate global journeys and cultural dialogues through carefully orchestrated spatial storytelling—each interior is an experience of travel distilled into a harmonious environment.",
        "image_url": "/images/designers/alessia_duval.jpg"
    },
    {
        "id": "adrian_mercer", 
        "name": "Adrian Mercer", 
        "description": "Transforms post-industrial materials into poetic, sculptural interiors full of moody drama.",
        "full_bio": "Emerging with striking presence on the design scene, Adrian Mercer is a fictitious visionary redefining 21st-century spatial aesthetics with bold yet refined sensibility. Mercer weaves narrative-rich intimacy into industrial materials, blending robust functionalism with poetic formality.\n\nDetailed Backstory of Adrian Mercer\n\nBorn in Glasgow in 1985, Mercer grew up amid the city's post-industrial shipyards and steel factories. His mother, a sculptor, and his father, who ran a metal-recycling yard, filled his childhood with oxidized copper plates, rusted gears, and battered steel fragments that seeded his future design language.\n\nAt Central Saint Martins, Mercer immersed himself in material experimentation and digital fabrication, fearlessly mixing traditional craft techniques with modern technology. Influenced by welded and molded industrial materials transformed into objects of refinement, he delved deeper into texture-driven sculptural aesthetics.\n\nAfter graduation, travels across Europe and Asia broadened his design vocabulary. Early projects—sculptural lamps, tables, and chairs—quickly raised his profile on returning to London.\n\nIn 2014, Mercer founded \"Mercer & Co.\" in a converted docklands warehouse, echoing his Glasgow roots. The studio became a playground for artisans, technologists, and sculptors, united by Mercer's credo: each project must tell a story, letting its industrial origins shine through even the most polished finishes.\n\nInfluences and Inspirations\n\nMercer draws on a broad constellation of references: Architects Tadao Ando and Zaha Hadid for material interplay and fluid dynamism. Brutalism and Cubism for angular forms, geometric abstraction, and raw textures. Global metalwork—from Moroccan hammered brassware to Japanese bronze sculpture—for tactile richness. Scotland's craggy coastlines, whose erosion patterns and oceanic geometry inspire his palettes of oxidized copper, tarnished silver, deep greys, and sea-blues.\n\nThis synthesis of industrial heritage, global craft, natural textures, and artistic boldness defines Mercer's signature—interiors that fuse comfort and luxury with rugged authenticity and sculptural drama.",
        "image_url": "/images/designers/adrian_mercer.jpg"
    },
    {
        "id": "lucien_hart", 
        "name": "Lucien Hart", 
        "description": "Fuses runway glamour with architectural audacity, delivering bold, theatrical spaces saturated in jewel tones.",
        "full_bio": "Origins\nBorn in Bordeaux (1985) to a fashion-stylist mother and architect father, Hart grew up scavenging ateliers and construction sites for inspiration. Summers in Paris revealed couture's backstage artistry; evenings at home taught him to sketch steel beams with the same reverence as silk drapery. Early on, he adopted a monochrome wardrobe: tailored cuts, vintage jewelry, quiet confidence.\n\nAt Politecnico di Milano and later the Royal College of Art, Hart blurred clothing and architecture—installations that wrapped spaces in fabric, rooms that wore shadows like veils. London's avant-garde circles took note; his graduation pieces—sculptural lamps and leather-clad lounges—sold before the paint dried.\n\nInspirations\nFashion: Alexander McQueen's narrative tension, Thierry Mugler's armored silhouettes, Tom Ford's sensual precision.\n\nFilm: Kubrick's unsettling rigor, Ridley Scott's dystopian density, Baz Luhrmann's decadent swoon.\n\nMovements: Vienna Secession unity, Art Deco exuberance.\n\nMaterials: Black marble, gilded steel, opalescent leather, digital light.\nThe result: interiors that feel like couture garments—structured, provocative, emotionally charged.\n\nStudio & Milestones\nHart launched Hart Atelier (London, with satellites in Paris and NYC) to craft \"spaces that breathe drama.\" Notable commissions include Maison Saint Laurent Flagship in Paris, \"Azure Twilight\" Residence in Corsica, and Nocturnal Hotel in New York.\n\nVision\nNow in his forties, Hart consults for houses seeking architectural drama. His credo: space is performance—luxury as rebellion, narrative as structure, emotion as material. By fusing couture audacity with built form, Lucien Hart continues to redraw the line between fashion and architecture.",
        "image_url": "/images/designers/lucien_hart.jpg"
    },
    {
        "id": "elinor_hartwell", 
        "name": "Elinor Hartwell", 
        "description": "Creates warm, soulful rooms where mindful living meets tactile, handcrafted comfort.",
        "full_bio": "Elinor Hartwell grew up in a windswept Devon village, where her family's seaside B&B and her mother's handmade pottery taught her that beauty lives in texture and imperfection. Mornings wandering salt-air gardens with her father, afternoons pressing clay—these rhythms shaped her intuitive grasp of material and mood.\n\nAfter studying interior architecture at the Royal College of Art, Hartwell honed her craft in Scandinavian studios, absorbing a refined simplicity that she melds with British warmth. Back in Brighton, she founded Hartwell Interiors, insisting every room feel \"genuinely lived in\": soft-edged alcoves for quiet reflection, communal nooks for laughter, surfaces that invite touch, not show.\n\nHer hallmark is a tactile palette—reclaimed timber, hand-woven linen, raw clay ceramics—layered in muted creams, sage greens, storm-cloud grays. She commissions local artisans to create pieces that bear fingerprints and tiny flaws, turning honesty into luxury.\n\nIn Cornwall's Cove Hotel, Hartwell blurred indoors with wave-tumbled stone and driftwood accents, crafting spaces that breathe with the sea breeze. At Sussex's Rosewood Retreat, she converted a Victorian farmhouse into a wellness haven, where organic linen drapes and sustainably sourced oak floors dissolve the line between architecture and landscape. Back home in Devon, The Hartwell Cottage—her childhood B&B reborn—melds vintage heirlooms with streamlined modernity, inviting guests into stories whispered over tea at a sunlit kitchen table.\n\nHartwell's designs feel less like décor and more like atmospheres—quiet sanctuaries that cultivate mindfulness, emotional ease, and the slow joy of simply being. In every project, she proves that true comfort arises when spaces honor the human spirit more than any trend.",
        "image_url": "/images/designers/elinor_hartwell.jpg"
    },
    {
        "id": "bianca_morelli", 
        "name": "Bianca Morelli", 
        "description": "Weaves fluid, organic forms and tactile layers into emotionally engaging, elegant interiors.",
        "full_bio": "Bianca Morelli channels the poetry of space into everyday ritual, crafting interiors that flow like watercolors—organic, tactile, and quietly transformative.\n\nRaised amid Florence's palaces and her mother's textile atelier, Morelli learned early that color and touch speak before words. Her father's architectural blueprints taught her form and function as a duet. Summers spent trailing through Renaissance gardens instilled a love for curves and layered light; afternoons at the loom revealed the magic of texture.\n\nAfter refining her vision at Politecnico di Milano, Morelli journeyed to Japan and Scandinavia, where she distilled minimalism into warmth and precision into ease. She absorbed the disciplined grace of Kyoto's temples and the soft functionality of Nordic fjord-side homes, forging a style both intimate and expansive—a harmony of restraint and emotional depth.\n\nIn her hands, materials awaken. Venetian glass sighs in undulating chandeliers; supple leather hugs sweeping benches; responsibly sourced timber blooms into furniture that feels less built than grown. Her palettes—muddy terracotta, river-stone gray, olive green laced with pearlescent accents—whisper of earth and sea, inviting touch and curiosity.\n\nAt the Luna Sea Residence in Positano, Morelli blurred walls and waves, folding terraces into living rooms with sinuous furniture that echoes ocean currents. Tokyo's Nimiko Spa Retreat became a sanctuary of stone and linen, where hand-carved river-rock basins and bamboo screens conjure meditative calm. In Oslo's Aurora Café and Lounge, fabric draperies ripple overhead like northern lights, while sculpted wooden tables foster gatherings both spontaneous and sacred.\n\nMorelli designs not to dazzle but to nurture. Each room unfolds like a lived story, a stage for quiet connection and emotional resonance. Her legacy is not mere decoration but atmospheres that heal, surprise, and endure—the art of making space itself feel like home.",
        "image_url": "/images/designers/bianca_morelli.jpg"
    },
    {
        "id": "eleanor_reed", 
        "name": "Eleanor Reed", 
        "description": "Mixes vintage patina with contemporary comfort for richly textured, eclectic authenticity.",
        "full_bio": "Inspired by the renowned design duo Roman & Williams, whose aesthetic juxtaposes timeless craftsmanship with surprising strokes of modernity, emerges the striking persona and vision of fictitious interior designer Eleanor Reed.\n\nEmergence and Early Life of Eleanor Reed\n\nBorn in the verdant countryside near Asheville, North Carolina, Eleanor Reed's bond with art and artisanship formed early in her childhood. For Eleanor, growing up was synonymous with afternoons spent wandering through antique markets alongside her woodworker father and painter mother. The slow rhythms of Appalachian tradition cultivated in her a deeply-rooted reverence for broader Southern crafts—pottery with irregular charm, meticulously hand-loomed fabrics, and cabinetry shaped by hand. As a child, Eleanor was captivated by contrasts. She found beauty in age-worn heirlooms reimagined against stark concrete walls, or classic furnishings juxtaposed against bright contemporary artworks.\n\nThroughout adolescence, Eleanor diversified her artistic portfolio, dabbling in sculpture, painting, and furniture restoration. Her fascination with objects' inherent stories evolved during family restoration efforts on their nineteenth-century home, experiences teaching her that 'imperfection is authenticity,' a mantra she carried forward. This philosophy spurred her initial interest in interiors: the understanding that spaces are collections of stories, each object a chapter that enriches life.\n\nEducation and Early Influences\n\nWith an emerging creativity keenly attuned to tactile beauty, Eleanor Reed pursued her education at the Rhode Island School of Design (RISD), a nurturing haven renowned for blending artistic excellence with purposeful reflection on materiality. There, Eleanor immersed herself fully in design theory, integrating sculptural forms, architecture fundamentals, sustainable approaches, and historical awareness into her approach.",
        "image_url": "/images/designers/eleanor_reed.jpg"
    },
    {
        "id": "oliver_renard", 
        "name": "Oliver Renard", 
        "description": "Stages maximalist fantasies with jewel-tone palettes, luxe textures, and theatrical storytelling.",
        "full_bio": "Oliver Renard arrives like a flourish of velvet drapery—his interiors unfold as immersive narratives, where color, texture, and scale dance in theatrical harmony. He designs not rooms but stages, each scene charged with drama, whimsy, and unapologetic grandeur.\n\nOrigins and Early Passion\nBorn in Charleston to a set-designer mother and playwright father, Renard spent childhood summers amid folding velvet curtains and hand-painted backdrops. Antique-flocked wallpaper whispered secrets of bygone elegance; ornate gardens rehearsed him in symmetry and surprise. He learned that every prop mattered—and that a single chandelier could steal the scene.\n\nFormative Training\nAt Parsons, Renard fused technical rigor with drama workshops, then crossed the Atlantic to London's drama school for a semester steeped in stagecraft. Those years taught him to choreograph light and shadow, to script a room's emotional arc, to make furniture feel like leading actors.\n\nSignature Aesthetic\nRenard's interiors hum with jewel-toned palettes—emerald booths, ruby drapes, cerulean walls—that pulse against gilded accents and antique heirlooms. Plush velvets collide with lacquered panels; sculptural fixtures hover like stage props. He revels in scale shifts: a low-slung sofa framed by soaring ceiling coffers, a tiny cabaret chair tucked beneath an over-the-top crystal chandelier. Nothing is timid; every element demands attention.\n\nNarrative-Driven Projects\nIn Charleston's Magnolia Hotel, each guest room unfolds like a chapter: butterfly-patterned murals, brass-bound four-post beds, banquettes draped in opulent silks. At New Orleans's Le Carnaval, Renard summoned Mardi Gras exuberance indoors—wings of peacock-blue velvet, gilded mirrors riffing on parade floats, bar stools upholstered in carnival feathers.\n\nFinal Act\nOliver Renard doesn't just decorate; he directs. His interiors are scripts you inhabit, performances you live. In his hands, a room becomes a rendezvous with wonder—a bold, vibrant testament to the power of design to captivate, delight, and transport.",
        "image_url": "/images/designers/oliver_renard.jpg"
    },
    {
        "id": "gabrielle_marlowe", 
        "name": "Gabrielle Marlowe", 
        "description": "Blends Southern graciousness with European classicism to craft airy, refined spaces of quiet luxury.",
        "full_bio": "Gabrielle Marlowe sculpts spaces where classical poise meets contemporary clarity—rooms that breathe Southern warmth and European refinement in equal measure.\n\nOrigins & Vision\nRaised among Savannah's antebellum portraits and her parents' historic archives, Gabrielle learned that every cornice and curve carries a story. Childhood summers in Provençal villas and Tuscan palazzos taught her to weave tradition into light-filled interiors.\n\nFormative Craft\nAt RISD, she mastered spatial choreography—lighting that feels like dawn, palettes drawn from olive groves and sea-sprayed shores, textures that invite the hand. Apprenticeships in New York townhouses and Atlanta's high-rise penthouses honed her gift for honoring original architecture while injecting fresh vitality.\n\nSignature Language\nHer rooms unfold in soft neutrals—ivory, stone gray, warm cocoa—with judicious notes of deep blue or emerald. She pairs crisp linens and plush velvets with polished woods and brushed metals. Symmetry anchors each vignette; unexpected accents—a sleek contemporary console, a sculptural lamp—spark intrigue without noise.\n\nStandout Commissions\nBellemeade Estate, Savannah: A Georgian mansion reborn—silken draperies framing antique moldings, deep-blue sofas floating atop reclaimed-wood floors, gilded sconces beside modern abstracts.\n\nLegacy\nGabrielle Marlowe's interiors feel both timeless and of-the-moment—a seamless dialogue between past and present. Each project reveals her talent for crafting spaces that look effortless yet are underpinned by fierce attention to detail and a deep respect for the stories buildings hold.",
        "image_url": "/images/designers/gabrielle_marlowe.jpg"
    },
    {
        "id": "elise_marceau", 
        "name": "Elise Marceau", 
        "description": "Balances minimalist restraint with tactile warmth, creating zen-like sanctuaries of European elegance.",
        "full_bio": "Elise Marceau sculpts serene sanctuaries where material honesty and quiet elegance converge.\n\nOrigins & Vision\nBorn amid Toulouse's medieval streets, she learned texture from her leather-artisan father and composition from her painter mother. Summers wandering Provençal ateliers and Spanish ceramics workshops taught her to see every object as a storyteller.\n\nCraft & Language\nTrained at École Camondo and refined under Milan's luxury ateliers, Marceau balances minimal layouts with sensorial richness: hand-glazed ceramics, softly woven linens, raw oak, and natural stone. Her palette—muted taupes, warm ivories, soft grays—whispers calm, while curves and layered textures invite touch.\n\nSignature Projects\nChâteau de Lumière, Provence: Original stone arches frame streamlined furnishings in ivory and linen. Custom ceramic pendants by local artisans become luminous focal points; open shelving displays curated travel mementos like gallery vignettes.\n\nTokyo Tranquility Hotel: Twenty private suites merge European restraint with Japanese \"Ma\"—spaces defined by intentional emptiness. Tactile wall finishes, Kyoto-crafted woodwork, and diffused rice-paper lighting cocoon guests in meditative comfort.\n\nEssence\nMarceau's rooms never shout—yet they linger in memory. She crafts environments that feel both grounded and poetic, honoring heritage while embracing modern clarity. In her world, less is luminous, and every surface tells a tale of craftsmanship, warmth, and timeless refinement.",
        "image_url": "/images/designers/elise_marceau.jpg"
    },
    {
        "id": "alexander_bennett", 
        "name": "Alexander Bennett", 
        "description": "Revives classical grandeur with tailored American sophistication and rich architectural detailing.",
        "full_bio": "Alexander Bennett: Classic Grandeur Meets Refined American Splendor\n\nRenowned for masterfully fusing classical architectural discipline with distinctly American luxury. Bennett's interiors present elegant proportions, harmonious compositions, and meticulous craftsmanship as vital elements of quality. Rich materials, historical motifs, and sophisticated combinations of modern and antique pieces characterize his spaces.\n\nBackstory: Early Life & Education\nBorn in Charleston, South Carolina, Alexander Bennett grew up immersed in historical charm and tradition. His father, an eminent architecture historian, specialized in colonial and neoclassical structures, while his mother, an antiques collector, frequented estate sales and gallery openings, passionately introducing young Alexander to sumptuous textures and timeless objects.\n\nRecognizing his son's talent, Bennett's father frequently invited Alexander into his lectures, teaching him discernment of key architectural attributes such as molding, symmetry, axial alignment, and proportional clarity. Alexander's profound understanding of historical accuracy and classical proportion became evident even as a teenager, exhibiting an exceptional ability to recreate period-oriented room settings blending authenticity with imaginative originality.\n\nPursuing this passion formally, Bennett studied Architecture and Decorative Arts at the Rhode Island School of Design, followed by advanced studies at Parsons School of Design in New York. During his impactful internship in Paris under renowned architect-designers specializing in classical restoration, Bennett gained vivid insights into rigorous architectural precision, fostering his passion for thoughtful, historically respectful articulation of space.\n\nConclusion\nToday, Alexander Bennett stands distinctly in the field of interior design, shaped by classical architectural precision married seamlessly with sumptuous American-inspired splendor. His exceptional talent marries architectural precision seamlessly with sumptuous American-inspired splendor, consistently breathing invigorating new life into historical relevance.",
        "image_url": "/images/designers/alexander_bennett.jpg"
    },
    {
        "id": "allegra_marquez", 
        "name": "Allegra Marquez", 
        "description": "Combines cultural authenticity with modern lines, marrying vibrant heritage motifs to Scandinavian restraint.",
        "full_bio": "Consistently champions spaces infused with cultural authenticity, tactile richness, and aesthetic serenity. Her hallmark lies in her ability to harmoniously blend worldly influences: layered textiles from Morocco, intricate Indian motifs, handcrafted ceramics from Japan, rich colors inspired by Latin American culture, and minimalistic lines evocative of Scandinavian interior design principles.\n\nIntroducing Allegra Marquez: Early Life and Background\nAllegra Marquez was born in Valencia, Spain, a city known for its vibrant festivals, dynamic history, magnificent architecture, and pottery traditions. Raised in an artistic family with a ceramicist mother and a father who was an antiques merchant, Allegra grew up surrounded by diverse cultural artifacts and creative traditions. Childhood holidays traversing through Portugal, Morocco, and Italy further immersed Allegra in the complexities, patterns, and textures of varying cultures and architectural legacies.\n\nEducational Pursuits and Early Influences\nDetermined to translate these inspirations professionally, Allegra studied interior architecture at the prestigious Politecnico di Milano, an academic hub celebrated for nurturing deeply innovative and culturally sensitive designers. Her rigorous education in Milan was a defining baptism into elegance, craftsmanship, and functional sensibilities informed by Italian modernism. However, Allegra's creative authenticity flourished outside the studio, traveling to traditional ethnic communities globally.\n\nCareer Milestones and Unique Influences\nAllegra's signature style quickly matured, characterized by the captivating eclecticism of global craftwork meshed integrally with minimalist modernity—she masterfully balanced vibrant global patterns with a pared-back, mindful simplicity. Sustainable sourcing became integral to her philosophy, directly tracing her experiences with local craftspeople; Allegra insisted on ethical provenance, believing interiors ought to speak not merely of sophistication but of integrity, mindful consumption, and respect for global cultures.",
        "image_url": "/images/designers/allegra_marquez.jpg"
    },
    {
        "id": "olivia_bennett", 
        "name": "Olivia Bennett", 
        "description": "Creates approachable elegance through thoughtful styling and sustainable, handcrafted details.",
        "full_bio": "Biography and Early Life:\n\nOlivia Bennett was born on a gentle spring morning in the picturesque countryside near Fredericton, New Brunswick. Growing up as the youngest of four siblings in a cozy, century-old farmhouse, Olivia's formative years were steeped in creative exploration, family warmth, and a genuine love for nature and heritage spaces. Her mother, a local artist, often encouraged playful creativity, allowing Olivia's inherent sensibility for color and aesthetics to emerge naturally.\n\nEducation and Career Development:\n\nFollowing secondary school, Olivia's affinity for the interplay of space, texture, and color drove her to pursue studies at Ryerson University's acclaimed School of Interior Design in Toronto. During her studies, Olson quickly established herself as a promising talent, frequently praised for innovative solutions that married functionality with bright and inviting aesthetics.\n\nAfter graduation, Olivia interned with notable designers in both Toronto and New York, eventually working alongside senior designers on notable boutique hotel redevelopments. Her experiences expanded her vision and deepened her appreciation for human-centric, welcoming interiors. Yet the restrained elegance and sometimes overly stylized atmosphere she saw during this period left her yearning for a more personalized, accessible approach to design.\n\nSpurred by an urge to carve her own path, Olivia established her studio \"Bennett Home Interiors\" in Prince Edward County, Ontario. The peaceful community, surrounded by rural charm, vineyards, and water views, inspired her to nurture a design approach rooted in relaxed elegance and livable beauty.",
        "image_url": "/images/designers/olivia_bennett.jpg"
    }
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

class StatusUpdate(BaseModel):
    status: str
    processed_image_url: Optional[str] = None
    error_message: Optional[str] = None
    prediction_id: Optional[str] = None

# User Management Models
class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    is_active: bool = True

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    referral_code: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(UserBase):
    id: str
    credits: int = 100  # Free tier starts with 100 credits
    subscription_status: str = "free"  # free, active, cancelled, expired
    subscription_plan: Optional[str] = None  # basic, pro, agency
    referral_code: str
    referred_by: Optional[str] = None
    total_referrals: int = 0
    created_at: datetime
    last_login: Optional[datetime] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    user: User

# Tool Configuration Models
class ToolRate(BaseModel):
    tool_name: str
    credits_per_use: int
    description: str

# Listing/Property Models
class PropertyDetails(BaseModel):
    address: str
    city: str
    state: str
    zip_code: str
    beds: int
    baths: float
    sqft: Optional[int] = None
    lot_size_sqft: Optional[int] = None
    year_built: Optional[int] = None
    property_type: str  # Single Family, Condo, Townhouse, etc.
    listing_price: Optional[float] = None
    mls_number: Optional[str] = None

class ListingPhoto(BaseModel):
    id: str
    filename: str
    url: str
    caption: Optional[str] = None
    is_primary: bool = False
    room_type: Optional[str] = None
    watermarked: bool = False
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    file_size: Optional[int] = None
    dimensions: Optional[Dict[str, int]] = None  # {width, height}

class InteriorDesignVariant(BaseModel):
    id: str
    original_image_id: str
    processed_image_url: str
    designer: str
    color_scheme: str
    room_type: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    watermarked: bool = False

class ModuleContent(BaseModel):
    content: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    last_edited: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    is_ai_generated: bool = True

class ChatMessage(BaseModel):
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    credits_used: int = 0

class AIToolSelection(BaseModel):
    tool_id: str
    tool_name: str
    category: str
    credits_cost: int
    selected: bool = False
    completed: bool = False
    output_data: Optional[Dict[str, Any]] = None

class Listing(BaseModel):
    id: str
    user_id: str
    property_details: PropertyDetails
    description: Optional[str] = None
    photos: List[ListingPhoto] = []
    interior_design_variants: List[InteriorDesignVariant] = []  # Processed interior design images
    selected_ai_tools: List[AIToolSelection] = []
    interior_designs: List[str] = []  # IDs of associated interior designs
    status: str = "draft"  # draft, active, pending, sold
    ai_processing_status: str = "pending"  # pending, processing, completed, failed
    ai_output: Optional[Dict[str, Any]] = None
    module_outputs: Dict[str, ModuleContent] = {}  # Key: module name, Value: content
    chat_history: Dict[str, List[ChatMessage]] = {}  # Key: module name, Value: chat messages
    agent_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class CreateListingRequest(BaseModel):
    property_details: PropertyDetails
    description: Optional[str] = None
    selected_tool_ids: List[str] = []
    agent_notes: Optional[str] = None

class UpdateListingRequest(BaseModel):
    property_details: Optional[PropertyDetails] = None
    description: Optional[str] = None
    selected_tool_ids: Optional[List[str]] = None
    status: Optional[str] = None
    agent_notes: Optional[str] = None

class GenerateModuleRequest(BaseModel):
    module_name: str
    additional_context: Optional[str] = None

class UpdateModuleRequest(BaseModel):
    content: str

class ChatImproveRequest(BaseModel):
    message: str
    module_name: str

class ImageDesignSettings(BaseModel):
    image_id: str
    room_type: str = "living_room"
    designer: str = "alessia_duval"
    color_scheme: str = "glacial_muse"

class ProcessInteriorDesignRequest(BaseModel):
    images: List[ImageDesignSettings]  # Each image has its own settings

# Agent Branding Models
class AgentBranding(BaseModel):
    id: str
    user_id: str
    logo_url: Optional[str] = None
    watermark_position: str = "bottom-right"  # bottom-right, bottom-left, top-right, top-left, center
    watermark_opacity: float = 0.7
    brand_colors: Dict[str, str] = {}  # primary, secondary colors
    created_at: datetime
    updated_at: datetime

# Admin Models
class AdminUser(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    role: str = "admin"
    created_at: datetime

# Authentication endpoints
@api_router.post("/auth/register", response_model=Token)
async def register_user(user_data: UserCreate):
    """Register a new user"""
    try:
        # Check if user already exists
        existing_user = await db.users.find_one({"email": user_data.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Hash password
        hashed_password = get_password_hash(user_data.password)
        
        # Generate unique referral code
        referral_code = generate_referral_code()
        while await db.users.find_one({"referral_code": referral_code}):
            referral_code = generate_referral_code()
        
        # Create user
        user = {
            "id": str(uuid.uuid4()),
            "email": user_data.email,
            "full_name": user_data.full_name,
            "hashed_password": hashed_password,
            "is_active": True,
            "credits": 100,  # Free tier starts with 100 credits
            "subscription_status": "free",
            "subscription_plan": None,
            "referral_code": referral_code,
            "referred_by": None,
            "total_referrals": 0,
            "created_at": datetime.utcnow(),
            "last_login": None
        }
        
        # Handle referral if provided
        if user_data.referral_code:
            referrer = await db.users.find_one({"referral_code": user_data.referral_code})
            if referrer and referrer.get('subscription_status') in ['active']:  # Must be paying member
                user["referred_by"] = referrer["id"]
                user["credits"] += 100  # Extra 100 credits for being referred
                
                # Give referrer 100 credits
                await db.users.update_one(
                    {"id": referrer["id"]},
                    {
                        "$inc": {"credits": 100, "total_referrals": 1}
                    }
                )
        
        # Insert user
        await db.users.insert_one(user)
        
        # Create access token
        access_token = create_access_token(data={"sub": user["id"]})
        
        # Remove sensitive data for response
        user_response = User(**{k: v for k, v in user.items() if k != 'hashed_password'})
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user=user_response
        )
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@api_router.post("/auth/login", response_model=Token)
async def login_user(user_data: UserLogin):
    """Login user"""
    try:
        # Find user
        user = await db.users.find_one({"email": user_data.email})
        if not user or not verify_password(user_data.password, user["hashed_password"]):
            raise HTTPException(status_code=401, detail="Incorrect email or password")
        
        if not user.get("is_active", True):
            raise HTTPException(status_code=401, detail="Account is disabled")
        
        # Update last login
        await db.users.update_one(
            {"id": user["id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        # Create access token
        access_token = create_access_token(data={"sub": user["id"]})
        
        # Remove sensitive data for response
        user_response = User(**{k: v for k, v in user.items() if k != 'hashed_password'})
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

# Enhanced authentication function to support both JWT and session tokens
async def get_current_user_enhanced(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user with support for both JWT and session tokens"""
    try:
        token = credentials.credentials
        
        # First, try session token
        session = await db.user_sessions.find_one({
            "session_token": token,
            "expires_at": {"$gt": datetime.utcnow()}
        })
        
        if session:
            user = await db.users.find_one({"id": session["user_id"]})
            if user:
                return User(**{k: v for k, v in user.items() if k != 'hashed_password'})
        
        # Fall back to JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
        user = await db.users.find_one({"id": user_id})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return User(**{k: v for k, v in user.items() if k != 'hashed_password'})
        
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")

@api_router.get("/auth/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user_enhanced)):
    """Get current user information"""
    return current_user

@api_router.get("/auth/credits")
async def get_user_credits(current_user: User = Depends(get_current_user_enhanced)):
    """Get user's current credit balance"""
    return {"credits": current_user.credits, "subscription_status": current_user.subscription_status}

# Google OAuth Session Models
class GoogleSessionRequest(BaseModel):
    user_data: Dict[str, Any]
    session_token: str

class ProcessSessionRequest(BaseModel):
    session_id: str

class UserSession(BaseModel):
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime

# Google OAuth endpoints
@api_router.post("/auth/google/session")
async def handle_google_session(session_data: GoogleSessionRequest):
    """Handle Google OAuth session and store user"""
    try:
        user_data = session_data.user_data
        session_token = session_data.session_token
        
        # Check if user exists by email
        existing_user = await db.users.find_one({"email": user_data["email"]})
        
        if existing_user:
            # User exists, update session
            user_id = existing_user["id"]
            user_response = User(**{k: v for k, v in existing_user.items() if k != 'hashed_password'})
        else:
            # Create new user for Google OAuth
            user_id = str(uuid.uuid4())
            referral_code = generate_referral_code()
            
            # Ensure unique referral code
            while await db.users.find_one({"referral_code": referral_code}):
                referral_code = generate_referral_code()
            
            new_user = {
                "id": user_id,
                "email": user_data["email"],
                "full_name": user_data.get("name", ""),
                "is_active": True,
                "credits": 100,  # Free tier starts with 100 credits
                "subscription_status": "free",
                "subscription_plan": None,
                "referral_code": referral_code,
                "referred_by": None,
                "total_referrals": 0,
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                "google_id": user_data.get("id"),
                "profile_picture": user_data.get("picture")
            }
            
            await db.users.insert_one(new_user)
            user_response = User(**{k: v for k, v in new_user.items() if k not in ['hashed_password', 'google_id', 'profile_picture']})
        
        # Store session token with 7-day expiry
        session_expires = datetime.utcnow() + timedelta(days=7)
        
        # Remove any existing sessions for this user
        await db.user_sessions.delete_many({"user_id": user_id})
        
        # Create new session
        session_doc = {
            "user_id": user_id,
            "session_token": session_token,
            "expires_at": session_expires,
            "created_at": datetime.utcnow()
        }
        
        await db.user_sessions.insert_one(session_doc)
        
        return {"success": True, "user": user_response}
        
    except Exception as e:
        logger.error(f"Google session handling error: {str(e)}")
        raise HTTPException(status_code=500, detail="Session handling failed")

@api_router.post("/auth/google/process-session")
async def process_google_oauth_session(session_request: ProcessSessionRequest):
    """Process Google OAuth session by calling Emergent OAuth service and storing user"""
    try:
        session_id = session_request.session_id
        logger.info(f"Processing Google OAuth session ID: {session_id}")
        
        # Call Emergent OAuth service to get session data
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
                headers={'X-Session-ID': session_id}
            ) as response:
                if response.status != 200:
                    logger.error(f"Emergent OAuth service error: {response.status}")
                    raise HTTPException(status_code=400, detail="Invalid session ID or OAuth service error")
                
                session_data = await response.json()
                
        logger.info(f"Retrieved OAuth session data for user: {session_data.get('email')}")
        
        # Extract user data and session token
        session_token = session_data.get('session_token')
        user_data = {k: v for k, v in session_data.items() if k != 'session_token'}
        
        if not session_token:
            raise HTTPException(status_code=400, detail="No session token in OAuth data")
        
        # Check if user exists by email
        existing_user = await db.users.find_one({"email": user_data["email"]})
        
        if existing_user:
            # User exists, update session
            user_id = existing_user["id"]
            user_response = User(**{k: v for k, v in existing_user.items() if k != 'hashed_password'})
        else:
            # Create new user for Google OAuth
            user_id = str(uuid.uuid4())
            referral_code = generate_referral_code()
            
            # Ensure unique referral code
            while await db.users.find_one({"referral_code": referral_code}):
                referral_code = generate_referral_code()
            
            new_user = {
                "id": user_id,
                "email": user_data["email"],
                "full_name": user_data.get("name", ""),
                "is_active": True,
                "credits": 100,  # Free tier starts with 100 credits
                "subscription_status": "free",
                "subscription_plan": None,
                "referral_code": referral_code,
                "referred_by": None,
                "total_referrals": 0,
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                "google_id": user_data.get("id"),
                "profile_picture": user_data.get("picture")
            }
            
            await db.users.insert_one(new_user)
            user_response = User(**{k: v for k, v in new_user.items() if k not in ['hashed_password', 'google_id', 'profile_picture']})
            logger.info(f"Created new user via Google OAuth: {user_data['email']}")
        
        # Store session token with 7-day expiry
        session_expires = datetime.utcnow() + timedelta(days=7)
        
        # Remove any existing sessions for this user
        await db.user_sessions.delete_many({"user_id": user_id})
        
        # Create new session
        session_doc = {
            "user_id": user_id,
            "session_token": session_token,
            "expires_at": session_expires,
            "created_at": datetime.utcnow()
        }
        
        await db.user_sessions.insert_one(session_doc)
        
        logger.info(f"Google OAuth processing complete for user: {user_data['email']}")
        
        return {
            "success": True, 
            "user": user_response,
            "session_token": session_token
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth session processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="OAuth session processing failed")

@api_router.post("/auth/logout")
async def logout_user(current_user: User = Depends(get_current_user_enhanced)):
    """Logout user and clear session"""
    try:
        # Delete all sessions for this user
        await db.user_sessions.delete_many({"user_id": current_user.id})
        return {"success": True, "message": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(status_code=500, detail="Logout failed")

# Listing Management Endpoints
@api_router.post("/listings", response_model=Listing)
async def create_listing(
    listing_data: CreateListingRequest,
    current_user: User = Depends(get_current_user_enhanced)
):
    """Create a new property listing"""
    try:
        listing_id = str(uuid.uuid4())
        
        # Convert selected tool IDs to AIToolSelection objects
        selected_tools = []
        for tool_id in listing_data.selected_tool_ids:
            if tool_id in AI_TOOLS_CATALOG:
                tool_info = AI_TOOLS_CATALOG[tool_id]
                selected_tools.append(AIToolSelection(
                    tool_id=tool_id,
                    tool_name=tool_info["name"],
                    category=tool_info["category"],
                    credits_cost=tool_info["credits_cost"],
                    selected=True
                ))
        
        now = datetime.utcnow()
        
        new_listing = {
            "id": listing_id,
            "user_id": current_user.id,
            "property_details": listing_data.property_details.dict(),
            "description": listing_data.description,
            "photos": [],
            "selected_ai_tools": [tool.dict() for tool in selected_tools],
            "interior_designs": [],
            "status": "draft",
            "ai_processing_status": "pending" if selected_tools else "completed",
            "ai_output": None,
            "agent_notes": listing_data.agent_notes,
            "created_at": now,
            "updated_at": now
        }
        
        await db.listings.insert_one(new_listing)
        
        # Convert back to Pydantic model for response
        return Listing(**new_listing)
        
    except Exception as e:
        logger.error(f"Create listing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create listing")

@api_router.get("/listings", response_model=List[Listing])
async def get_user_listings(
    current_user: User = Depends(get_current_user_enhanced)
):
    """Get all listings for the current user"""
    try:
        listings_cursor = db.listings.find({"user_id": current_user.id}).sort("created_at", -1)
        listings = await listings_cursor.to_list(length=None)
        
        return [Listing(**{k: v for k, v in listing.items() if k != '_id'}) for listing in listings]
        
    except Exception as e:
        logger.error(f"Get listings error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch listings")

@api_router.get("/listings/{listing_id}", response_model=Listing)
async def get_listing(
    listing_id: str,
    current_user: User = Depends(get_current_user_enhanced)
):
    """Get a specific listing"""
    try:
        listing = await db.listings.find_one({
            "id": listing_id,
            "user_id": current_user.id
        })
        
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        return Listing(**{k: v for k, v in listing.items() if k != '_id'})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get listing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch listing")

@api_router.put("/listings/{listing_id}", response_model=Listing)
async def update_listing(
    listing_id: str,
    update_data: UpdateListingRequest,
    current_user: User = Depends(get_current_user_enhanced)
):
    """Update a listing"""
    try:
        # Check if listing exists and belongs to user
        existing_listing = await db.listings.find_one({
            "id": listing_id,
            "user_id": current_user.id
        })
        
        if not existing_listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        # Build update document
        update_doc = {"updated_at": datetime.utcnow()}
        
        if update_data.property_details:
            update_doc["property_details"] = update_data.property_details.dict()
        if update_data.description is not None:
            update_doc["description"] = update_data.description
        if update_data.status:
            update_doc["status"] = update_data.status
        if update_data.agent_notes is not None:
            update_doc["agent_notes"] = update_data.agent_notes
        
        if update_data.selected_tool_ids is not None:
            # Update selected AI tools
            selected_tools = []
            for tool_id in update_data.selected_tool_ids:
                if tool_id in AI_TOOLS_CATALOG:
                    tool_info = AI_TOOLS_CATALOG[tool_id]
                    selected_tools.append(AIToolSelection(
                        tool_id=tool_id,
                        tool_name=tool_info["name"],
                        category=tool_info["category"],
                        credits_cost=tool_info["credits_cost"],
                        selected=True
                    ))
            update_doc["selected_ai_tools"] = [tool.dict() for tool in selected_tools]
            update_doc["ai_processing_status"] = "pending" if selected_tools else "completed"
        
        # Update the listing
        await db.listings.update_one(
            {"id": listing_id, "user_id": current_user.id},
            {"$set": update_doc}
        )
        
        # Fetch and return updated listing
        updated_listing = await db.listings.find_one({
            "id": listing_id,
            "user_id": current_user.id
        })
        
        return Listing(**{k: v for k, v in updated_listing.items() if k != '_id'})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update listing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update listing")

@api_router.delete("/listings/{listing_id}")
async def delete_listing(
    listing_id: str,
    current_user: User = Depends(get_current_user_enhanced)
):
    """Delete a listing"""
    try:
        result = await db.listings.delete_one({
            "id": listing_id,
            "user_id": current_user.id
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        return {"success": True, "message": "Listing deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete listing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete listing")

# AI Tools Endpoints
@api_router.get("/ai-tools")
async def get_ai_tools():
    """Get available AI tools catalog"""
    try:
        # Group tools by category
        tools_by_category = {}
        for tool_id, tool_info in AI_TOOLS_CATALOG.items():
            category = tool_info["category"]
            if category not in tools_by_category:
                tools_by_category[category] = []
            tools_by_category[category].append(tool_info)
        
        return {
            "tools_by_category": tools_by_category,
            "total_tools": len(AI_TOOLS_CATALOG)
        }
        
    except Exception as e:
        logger.error(f"Get AI tools error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch AI tools")

# MCP Mega-Agent Processing
@api_router.post("/listings/{listing_id}/process-ai")
async def process_listing_ai_tools(
    listing_id: str,
    current_user: User = Depends(get_current_user_enhanced)
):
    """Process all selected AI tools for a listing using the mega-agent"""
    try:
        # Get the listing
        listing = await db.listings.find_one({
            "id": listing_id,
            "user_id": current_user.id
        })
        
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        # Check if there are selected tools
        selected_tools = listing.get("selected_ai_tools", [])
        if not selected_tools:
            raise HTTPException(status_code=400, detail="No AI tools selected for this listing")
        
        # Calculate total credits needed
        total_credits_needed = sum(tool.get("credits_cost", 0) for tool in selected_tools)
        
        # Check if user has enough credits
        if current_user.credits < total_credits_needed:
            raise HTTPException(
                status_code=402, 
                detail=f"Insufficient credits. Need {total_credits_needed}, have {current_user.credits}"
            )
        
        # Update listing status to processing
        await db.listings.update_one(
            {"id": listing_id, "user_id": current_user.id},
            {
                "$set": {
                    "ai_processing_status": "processing",
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Import and run the mega-agent
        from mcp_agent_server import run_mega_agent_processing
        
        # Process with mega-agent
        processing_result = await run_mega_agent_processing(listing, selected_tools)
        
        if processing_result.get('success'):
            # Deduct credits from user
            await db.users.update_one(
                {"id": current_user.id},
                {"$inc": {"credits": -total_credits_needed}}
            )
            
            # Update listing with results
            await db.listings.update_one(
                {"id": listing_id, "user_id": current_user.id},
                {
                    "$set": {
                        "ai_processing_status": "completed",
                        "ai_output": processing_result,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            # Mark all tools as completed
            updated_tools = []
            for tool in selected_tools:
                tool["completed"] = True
                tool["processed_at"] = datetime.utcnow().isoformat()
                updated_tools.append(tool)
            
            await db.listings.update_one(
                {"id": listing_id, "user_id": current_user.id},
                {"$set": {"selected_ai_tools": updated_tools}}
            )
            
            return {
                "success": True,
                "processing_id": processing_result.get("processing_id"),
                "tools_processed": len(selected_tools),
                "credits_used": total_credits_needed,
                "remaining_credits": current_user.credits - total_credits_needed,
                "message": "AI processing completed successfully"
            }
        else:
            # Update listing status to failed
            await db.listings.update_one(
                {"id": listing_id, "user_id": current_user.id},
                {
                    "$set": {
                        "ai_processing_status": "failed",
                        "ai_output": processing_result,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            raise HTTPException(status_code=500, detail="AI processing failed")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI processing error: {str(e)}")
        
        # Update listing status to failed
        try:
            await db.listings.update_one(
                {"id": listing_id, "user_id": current_user.id},
                {
                    "$set": {
                        "ai_processing_status": "failed",
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        except:
            pass
            
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")

@api_router.get("/listings/{listing_id}/ai-results")
async def get_listing_ai_results(
    listing_id: str,
    current_user: User = Depends(get_current_user_enhanced)
):
    """Get AI processing results for a listing"""
    try:
        listing = await db.listings.find_one({
            "id": listing_id,
            "user_id": current_user.id
        })
        
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        ai_output = listing.get("ai_output")
        if not ai_output:
            raise HTTPException(status_code=404, detail="No AI results found for this listing")
        
        return {
            "listing_id": listing_id,
            "processing_status": listing.get("ai_processing_status"),
            "ai_results": ai_output,
            "tools_processed": listing.get("selected_ai_tools", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get AI results error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch AI results")


# Image Upload & Management for Listings
@api_router.post("/listings/{listing_id}/images/upload")
async def upload_listing_images(
    listing_id: str,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user_enhanced)
):
    """Upload multiple images for a listing"""
    try:
        # Verify listing ownership
        listing = await db.listings.find_one({
            "id": listing_id,
            "user_id": current_user.id
        })
        
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        # Create listing images directory
        images_dir = PROCESSED_IMAGES_DIR / "listings" / listing_id
        images_dir.mkdir(parents=True, exist_ok=True)
        
        uploaded_images = []
        
        for file in files:
            # Validate file type
            if not file.content_type.startswith('image/'):
                continue  # Skip non-image files
            
            # Generate unique filename
            file_extension = Path(file.filename).suffix.lower()
            image_id = str(uuid.uuid4())
            filename = f"{image_id}{file_extension}"
            file_path = images_dir / filename
            
            # Save the file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Create photo record
            photo = {
                "id": image_id,
                "filename": filename,
                "url": f"/api/listings/{listing_id}/images/{filename}",
                "caption": None,
                "is_primary": len(listing.get("photos", [])) == 0,  # First image is primary
                "room_type": None,
                "watermarked": False,
                "uploaded_at": datetime.utcnow(),
                "file_size": len(content)
            }
            
            uploaded_images.append(photo)
        
        # Update listing with new photos
        await db.listings.update_one(
            {"id": listing_id},
            {
                "$push": {"photos": {"$each": uploaded_images}},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return {
            "success": True,
            "uploaded_count": len(uploaded_images),
            "images": uploaded_images
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload images: {str(e)}")

@api_router.get("/listings/{listing_id}/images")
async def get_listing_images(
    listing_id: str,
    current_user: User = Depends(get_current_user_enhanced)
):
    """Get all images for a listing"""
    try:
        listing = await db.listings.find_one({
            "id": listing_id,
            "user_id": current_user.id
        })
        
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        return {
            "listing_id": listing_id,
            "photos": listing.get("photos", []),
            "interior_design_variants": listing.get("interior_design_variants", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get images error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch images")

@api_router.get("/listings/{listing_id}/images/{filename}")
async def serve_listing_image(
    listing_id: str,
    filename: str
):
    """Serve a listing image file"""
    try:
        file_path = PROCESSED_IMAGES_DIR / "listings" / listing_id / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(
            path=file_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=31536000"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Serve image error: {str(e)}")
        raise HTTPException(status_code=404, detail="Image not found")

@api_router.delete("/listings/{listing_id}/images/{image_id}")
async def delete_listing_image(
    listing_id: str,
    image_id: str,
    current_user: User = Depends(get_current_user_enhanced)
):
    """Delete a listing image"""
    try:
        listing = await db.listings.find_one({
            "id": listing_id,
            "user_id": current_user.id
        })
        
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        # Find the image
        photos = listing.get("photos", [])
        image = next((p for p in photos if p["id"] == image_id), None)
        
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Delete file
        file_path = PROCESSED_IMAGES_DIR / "listings" / listing_id / image["filename"]
        if file_path.exists():
            file_path.unlink()
        
        # Remove from database
        await db.listings.update_one(
            {"id": listing_id},
            {
                "$pull": {"photos": {"id": image_id}},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return {"success": True, "message": "Image deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete image error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete image")

# Module Content Generation & Management
@api_router.post("/listings/{listing_id}/modules/{module_name}/generate")
async def generate_module_content(
    listing_id: str,
    module_name: str,
    request: GenerateModuleRequest,
    current_user: User = Depends(get_current_user_enhanced)
):
    """Generate AI content for a specific module"""
    try:
        # Verify listing ownership
        listing = await db.listings.find_one({
            "id": listing_id,
            "user_id": current_user.id
        })
        
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        # Check credits (1 credit per generation)
        if current_user.credits < 1:
            raise HTTPException(status_code=402, detail="Insufficient credits")
        
        # Deduct credits
        await db.users.update_one(
            {"id": current_user.id},
            {"$inc": {"credits": -1}}
        )
        
        # Prepare context from listing
        property_details = listing.get("property_details", {})
        context = f"""
Property Address: {property_details.get('address', 'N/A')}
City: {property_details.get('city', 'N/A')}, State: {property_details.get('state', 'N/A')}
Property Type: {property_details.get('property_type', 'N/A')}
Bedrooms: {property_details.get('beds', 'N/A')}
Bathrooms: {property_details.get('baths', 'N/A')}
Square Feet: {property_details.get('sqft', 'N/A')}
Listing Price: ${property_details.get('listing_price', 'TBD')}
Description: {listing.get('description', 'N/A')}

Additional Context: {request.additional_context or 'None provided'}
"""
        
        # Define module-specific prompts
        module_prompts = {
            "listing_copy": "Write a compelling, professional property listing description that highlights key features and creates buyer interest. Be specific and engaging.",
            "marketing_copy": "Create marketing copy for this property that can be used in emails, social media, and advertisements. Make it attention-grabbing and persuasive.",
            "social_media": "Generate 3 different social media posts for this property listing. Make them engaging, include relevant hashtags, and vary the tone.",
            "email_template": "Write a professional email template that a real estate agent can use to introduce this property to potential buyers. Include a compelling subject line.",
            "market_intel": "Provide market intelligence and competitive positioning for this property. Include pricing strategy recommendations and target buyer profile.",
            "virtual_tour_script": "Create a script for a virtual tour or property walkthrough video. Make it conversational and highlight unique selling points."
        }
        
        system_message = f"You are an expert real estate copywriter and marketing professional. Generate high-quality, professional content for real estate listings."
        prompt = f"{module_prompts.get(module_name, 'Generate professional content for this property listing.')}\n\nProperty Information:\n{context}"
        
        # Try with Emergent LLM key first, fall back to OpenAI key if it fails
        response = None
        api_key_used = None
        
        # First try: Emergent LLM key
        emergent_key = os.environ.get('EMERGENT_LLM_KEY')
        openai_key = os.environ.get('OPENAI_API_KEY')
        
        if emergent_key and emergent_key.startswith('sk-emergent'):
            try:
                logger.info(f"Attempting content generation with Emergent LLM key")
                chat = LlmChat(
                    api_key=emergent_key,
                    session_id=f"listing_{listing_id}_module_{module_name}",
                    system_message=system_message
                ).with_model("openai", "gpt-4o")
                
                user_message = UserMessage(text=prompt)
                response = await chat.send_message(user_message)
                api_key_used = "Emergent LLM"
                logger.info(f"✅ Content generated successfully with Emergent LLM key")
            except Exception as e:
                logger.warning(f"Emergent LLM key failed: {str(e)}, falling back to OpenAI key")
                response = None
        
        # Fallback to OpenAI key if Emergent failed or not available
        if not response and openai_key:
            try:
                logger.info(f"Attempting content generation with OpenAI API key (fallback)")
                chat = LlmChat(
                    api_key=openai_key,
                    session_id=f"listing_{listing_id}_module_{module_name}",
                    system_message=system_message
                ).with_model("openai", "gpt-4o")
                
                user_message = UserMessage(text=prompt)
                response = await chat.send_message(user_message)
                api_key_used = "OpenAI"
                logger.info(f"✅ Content generated successfully with OpenAI API key")
            except Exception as e:
                logger.error(f"OpenAI API key also failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to generate content with both API keys: {str(e)}")
        
        if not response:
            raise HTTPException(status_code=500, detail="No API key available or all keys failed")
        
        # Store module content
        module_content = {
            "content": response,
            "generated_at": datetime.utcnow(),
            "last_edited": datetime.utcnow(),
            "version": 1,
            "is_ai_generated": True
        }
        
        # Update listing
        await db.listings.update_one(
            {"id": listing_id},
            {
                "$set": {
                    f"module_outputs.{module_name}": module_content,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return {
            "success": True,
            "module_name": module_name,
            "content": response,
            "credits_used": 1,
            "remaining_credits": current_user.credits - 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generate module content error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate content: {str(e)}")

@api_router.put("/listings/{listing_id}/modules/{module_name}")
async def update_module_content(
    listing_id: str,
    module_name: str,
    request: UpdateModuleRequest,
    current_user: User = Depends(get_current_user_enhanced)
):
    """Update module content (manual edit)"""
    try:
        # Verify listing ownership
        listing = await db.listings.find_one({
            "id": listing_id,
            "user_id": current_user.id
        })
        
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        # Get existing module content
        module_outputs = listing.get("module_outputs", {})
        existing_module = module_outputs.get(module_name, {})
        
        # Update content
        updated_module = {
            "content": request.content,
            "generated_at": existing_module.get("generated_at", datetime.utcnow()),
            "last_edited": datetime.utcnow(),
            "version": existing_module.get("version", 1) + 1,
            "is_ai_generated": False
        }
        
        # Update listing
        await db.listings.update_one(
            {"id": listing_id},
            {
                "$set": {
                    f"module_outputs.{module_name}": updated_module,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return {
            "success": True,
            "module_name": module_name,
            "content": request.content
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update module content error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update content")

@api_router.post("/listings/{listing_id}/modules/{module_name}/chat")
async def chat_improve_module(
    listing_id: str,
    module_name: str,
    request: ChatImproveRequest,
    current_user: User = Depends(get_current_user_enhanced)
):
    """Chat with AI to improve module content (1 credit per message)"""
    try:
        # Verify listing ownership
        listing = await db.listings.find_one({
            "id": listing_id,
            "user_id": current_user.id
        })
        
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        # Check credits
        if current_user.credits < 1:
            raise HTTPException(status_code=402, detail="Insufficient credits")
        
        # Deduct credits
        await db.users.update_one(
            {"id": current_user.id},
            {"$inc": {"credits": -1}}
        )
        
        # Get current module content
        module_outputs = listing.get("module_outputs", {})
        current_content = module_outputs.get(module_name, {}).get("content", "")
        
        if not current_content:
            raise HTTPException(status_code=404, detail="No content found for this module. Generate content first.")
        
        # Get existing chat history
        chat_history = listing.get("chat_history", {}).get(module_name, [])
        
        # Build conversation context
        conversation = f"Current {module_name} content:\n\n{current_content}\n\nUser request: {request.message}"
        
        # Try with Emergent LLM key first, fall back to OpenAI key if it fails
        response = None
        api_key_used = None
        
        # First try: Emergent LLM key
        emergent_key = os.environ.get('EMERGENT_LLM_KEY')
        openai_key = os.environ.get('OPENAI_API_KEY')
        
        if emergent_key and emergent_key.startswith('sk-emergent'):
            try:
                logger.info(f"Attempting chat with Emergent LLM key")
                chat = LlmChat(
                    api_key=emergent_key,
                    session_id=f"listing_{listing_id}_chat_{module_name}",
                    system_message="You are a helpful assistant for improving real estate listing content. The user will provide feedback on existing content and you should help them refine it."
                ).with_model("openai", "gpt-4o")
                
                user_message = UserMessage(text=conversation)
                response = await chat.send_message(user_message)
                api_key_used = "Emergent LLM"
                logger.info(f"✅ Chat response generated successfully with Emergent LLM key")
            except Exception as e:
                logger.warning(f"Emergent LLM key failed: {str(e)}, falling back to OpenAI key")
                response = None
        
        # Fallback to OpenAI key if Emergent failed or not available
        if not response and openai_key:
            try:
                logger.info(f"Attempting chat with OpenAI API key (fallback)")
                chat = LlmChat(
                    api_key=openai_key,
                    session_id=f"listing_{listing_id}_chat_{module_name}",
                    system_message="You are a helpful assistant for improving real estate listing content. The user will provide feedback on existing content and you should help them refine it."
                ).with_model("openai", "gpt-4o")
                
                user_message = UserMessage(text=conversation)
                response = await chat.send_message(user_message)
                api_key_used = "OpenAI"
                logger.info(f"✅ Chat response generated successfully with OpenAI API key")
            except Exception as e:
                logger.error(f"OpenAI API key also failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to generate chat response with both API keys: {str(e)}")
        
        if not response:
            raise HTTPException(status_code=500, detail="No API key available or all keys failed")
        
        # Create chat messages
        user_chat_msg = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow(),
            "credits_used": 0
        }
        
        assistant_chat_msg = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow(),
            "credits_used": 1
        }
        
        # Append to chat history
        chat_history.extend([user_chat_msg, assistant_chat_msg])
        
        # Update listing with chat history
        await db.listings.update_one(
            {"id": listing_id},
            {
                "$set": {
                    f"chat_history.{module_name}": chat_history,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return {
            "success": True,
            "response": response,
            "credits_used": 1,
            "remaining_credits": current_user.credits - 1,
            "chat_history": chat_history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat improve error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")

# Interior Design Processing for Listing Images
@api_router.post("/listings/{listing_id}/interior-design/process")
async def process_listing_interior_design(
    listing_id: str,
    request: ProcessInteriorDesignRequest,
    current_user: User = Depends(get_current_user_enhanced)
):
    """Process selected listing images through interior design AI"""
    try:
        # Verify listing ownership
        listing = await db.listings.find_one({
            "id": listing_id,
            "user_id": current_user.id
        })
        
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        
        # Verify images exist
        photos = listing.get("photos", [])
        image_settings_map = {img.image_id: img for img in request.images}
        selected_photos = [p for p in photos if p["id"] in image_settings_map]
        
        if not selected_photos:
            raise HTTPException(status_code=404, detail="No valid images found")
        
        # Calculate credits (5 credits per image for interior design)
        credits_per_image = await get_tool_rate("interior_design")
        total_credits = credits_per_image * len(selected_photos)
        
        # Check credits
        if current_user.credits < total_credits:
            raise HTTPException(
                status_code=402,
                detail=f"Insufficient credits. Need {total_credits}, have {current_user.credits}"
            )
        
        # Deduct credits
        await db.users.update_one(
            {"id": current_user.id},
            {"$inc": {"credits": -total_credits}}
        )
        
        # Process each image with its own settings
        processed_variants = []
        
        for photo in selected_photos:
            settings = image_settings_map[photo["id"]]
            variant_id = str(uuid.uuid4())
            
            # Get the image file path
            image_path = PROCESSED_IMAGES_DIR / "listings" / listing_id / photo["filename"]
            
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                continue
            
            # Generate prompt for this image's settings
            generated_prompt = await generate_design_prompt_with_assistant(
                settings.room_type, 
                settings.designer, 
                settings.color_scheme
            )
            
            # Create design request record
            design_request = InteriorDesignRequest(
                original_filename=photo["filename"],
                status="queued",
                room_type=settings.room_type,
                designer=settings.designer,
                color_scheme=settings.color_scheme,
                generated_prompt=generated_prompt
            )
            
            design_dict = design_request.dict()
            design_dict["user_id"] = current_user.id
            design_dict["credits_used"] = credits_per_image
            design_dict["listing_id"] = listing_id
            design_dict["original_image_id"] = photo["id"]
            
            await db.interior_designs.insert_one(design_dict)
            
            # Process asynchronously
            import asyncio
            asyncio.create_task(process_image_async(
                design_request.id, 
                image_path, 
                generated_prompt, 
                photo["filename"], 
                settings.room_type, 
                settings.designer, 
                settings.color_scheme
            ))
            
            # Create variant record (will be updated when processing completes)
            variant = {
                "id": variant_id,
                "design_request_id": design_request.id,
                "original_image_id": photo["id"],
                "status": "processing",
                "designer": settings.designer,
                "color_scheme": settings.color_scheme,
                "room_type": settings.room_type,
                "created_at": datetime.utcnow()
            }
            
            processed_variants.append(variant)
        
        # Update listing with variants
        await db.listings.update_one(
            {"id": listing_id},
            {
                "$push": {"interior_design_variants": {"$each": processed_variants}},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return {
            "success": True,
            "processed_count": len(processed_variants),
            "credits_used": total_credits,
            "remaining_credits": current_user.credits - total_credits,
            "variants": processed_variants,
            "message": "Images are being processed. Check back in a few minutes."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process interior design error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process images: {str(e)}")

# Agent Branding & Watermarking Endpoints
@api_router.post("/branding/upload-logo")
async def upload_agent_logo(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user_enhanced)
):
    """Upload agent logo for watermarking"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create branding directory if it doesn't exist
        branding_dir = Path("storage/branding")
        branding_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_extension = Path(file.filename).suffix.lower()
        filename = f"{current_user.id}_logo{file_extension}"
        file_path = branding_dir / filename
        
        # Save the file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Update or create agent branding record
        branding_data = {
            "user_id": current_user.id,
            "logo_url": f"/api/branding/logo/{filename}",
            "watermark_position": "bottom-right",
            "watermark_opacity": 0.7,
            "brand_colors": {},
            "updated_at": datetime.utcnow()
        }
        
        # Check if branding already exists
        existing_branding = await db.agent_branding.find_one({"user_id": current_user.id})
        
        if existing_branding:
            await db.agent_branding.update_one(
                {"user_id": current_user.id},
                {"$set": branding_data}
            )
        else:
            branding_data["id"] = str(uuid.uuid4())
            branding_data["created_at"] = datetime.utcnow()
            await db.agent_branding.insert_one(branding_data)
        
        return {
            "success": True,
            "logo_url": branding_data["logo_url"],
            "message": "Logo uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logo upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload logo")

@api_router.get("/branding/logo/{filename}")
async def serve_agent_logo(filename: str):
    """Serve agent logo files"""
    try:
        file_path = Path("storage/branding") / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Logo not found")
        
        # Determine media type
        media_type = "image/jpeg"
        if filename.lower().endswith('.png'):
            media_type = "image/png"
        elif filename.lower().endswith('.gif'):
            media_type = "image/gif"
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            headers={
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": f"inline; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Serve logo error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to serve logo")

@api_router.get("/branding/settings", response_model=AgentBranding)
async def get_agent_branding(
    current_user: User = Depends(get_current_user_enhanced)
):
    """Get agent branding settings"""
    try:
        branding = await db.agent_branding.find_one({"user_id": current_user.id})
        
        if not branding:
            # Create default branding
            default_branding = {
                "id": str(uuid.uuid4()),
                "user_id": current_user.id,
                "logo_url": None,
                "watermark_position": "bottom-right",
                "watermark_opacity": 0.7,
                "brand_colors": {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            await db.agent_branding.insert_one(default_branding)
            return AgentBranding(**default_branding)
        
        return AgentBranding(**{k: v for k, v in branding.items() if k != '_id'})
        
    except Exception as e:
        logger.error(f"Get branding error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch branding settings")

class BrandingUpdateRequest(BaseModel):
    position: str = "bottom-right"
    opacity: float = 0.7
    brand_colors: Dict[str, str] = {}

@api_router.put("/branding/settings")
async def update_agent_branding(
    update_data: BrandingUpdateRequest,
    current_user: User = Depends(get_current_user_enhanced)
):
    """Update agent branding settings"""
    try:
        # Validate inputs
        valid_positions = ["bottom-right", "bottom-left", "top-right", "top-left", "center"]
        if update_data.position not in valid_positions:
            raise HTTPException(status_code=400, detail="Invalid watermark position")
        
        if not 0 <= update_data.opacity <= 1:
            raise HTTPException(status_code=400, detail="Opacity must be between 0 and 1")
        
        update_doc = {
            "watermark_position": update_data.position,
            "watermark_opacity": update_data.opacity,
            "brand_colors": update_data.brand_colors,
            "updated_at": datetime.utcnow()
        }
        
        result = await db.agent_branding.update_one(
            {"user_id": current_user.id},
            {"$set": update_doc}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Branding settings not found")
        
        return {"success": True, "message": "Branding settings updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update branding error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update branding settings")

# AI Tools Configuration
AI_TOOLS_CATALOG = {
    # Marketing & Creative
    "listing_luxe_gpt": {
        "id": "listing_luxe_gpt",
        "name": "Listing-Luxe GPT",
        "category": "Marketing & Creative",
        "description": "Rewrites raw bullet points into MLS-ready, SEO-savvy descriptions in the agent's chosen tone.",
        "credits_cost": 3,
        "inputs": ["room_details", "upgrades", "tone_slider"],
        "outputs": ["listing_description", "headline", "social_snippets"]
    },
    "social_snippets_studio": {
        "id": "social_snippets_studio", 
        "name": "Social-Snippets Studio",
        "category": "Marketing & Creative",
        "description": "Spits out a week of reels hooks, carousel copy, hashtag sets, and CTA captions per listing.",
        "credits_cost": 2,
        "inputs": ["listing_url", "tone", "platform_mix"],
        "outputs": ["ig_captions", "reels_scripts", "hashtag_clusters"]
    },
    "photofix_wizard": {
        "id": "photofix_wizard",
        "name": "PhotoFix Wizard", 
        "category": "Marketing & Creative",
        "description": "Detects poor listing photos, auto-suggests retouching notes and Stable Diffusion prompts.",
        "credits_cost": 2,
        "inputs": ["photos"],
        "outputs": ["before_after_mockups", "prompt_blocks"]
    },
    "walk_score_wordsmith": {
        "id": "walk_score_wordsmith",
        "name": "Walk-Score Wordsmith",
        "category": "Marketing & Creative", 
        "description": "Turns raw WalkScore/TransitScore/BikeScore data into marketing copy snippets.",
        "credits_cost": 1,
        "inputs": ["walk_score_data"],
        "outputs": ["headlines", "marketing_blurbs"]
    },
    "foreign_buyer_friendly": {
        "id": "foreign_buyer_friendly",
        "name": "Foreign-Buyer Friendly",
        "category": "Marketing & Creative",
        "description": "Instantly translates listings & marketing collateral into top five languages in your market.",
        "credits_cost": 4,
        "inputs": ["listing_copy"],
        "outputs": ["multilingual_pack"]
    },
    "video_tour_scriptsmith": {
        "id": "video_tour_scriptsmith",
        "name": "Video-Tour Scriptsmith",
        "category": "Marketing & Creative",
        "description": "Writes 60-second vertical-video scripts timed to pan shots, plus on-screen caption text.",
        "credits_cost": 3,
        "inputs": ["property_details", "video_style"],
        "outputs": ["storyboard", "teleprompter_script"]
    },
    
    # Staging & Design
    "staging_style_coach": {
        "id": "staging_style_coach",
        "name": "Staging Style Coach",
        "category": "Staging & Design",
        "description": "Suggests virtual-staging looks and outputs a ready-to-order brief for your staging vendor.",
        "credits_cost": 3,
        "inputs": ["photos", "floor_plan", "target_buyer_profile"],
        "outputs": ["mood_board", "furniture_list", "staging_prompts"]
    },
    
    # Due-Diligence & Compliance  
    "contract_clarifier": {
        "id": "contract_clarifier",
        "name": "Contract-Clarifier",
        "category": "Due-Diligence & Compliance",
        "description": "Translates state-specific contract clauses into plain English and highlights gotchas.",
        "credits_cost": 2,
        "inputs": ["contract_pdf", "state"],
        "outputs": ["simplified_text", "faq", "signature_checklist"]
    },
    "fair_housing_guard": {
        "id": "fair_housing_guard", 
        "name": "Fair-Housing Compliance Guard",
        "category": "Due-Diligence & Compliance",
        "description": "Scans any marketing copy or ad creative and flags potential Fair-Housing or local-regulation violations.",
        "credits_cost": 1,
        "inputs": ["ad_text", "images"],
        "outputs": ["violation_flags", "safe_rewrites"]
    },
    "permit_pulse": {
        "id": "permit_pulse",
        "name": "PermitPulse", 
        "category": "Due-Diligence & Compliance",
        "description": "Pulls local building-permit history & flags red-flag renovations or missing finals.",
        "credits_cost": 3,
        "inputs": ["property_address"],
        "outputs": ["permit_timeline", "inspection_checklist"]
    },
    "flood_fire_radar": {
        "id": "flood_fire_radar",
        "name": "Flood & Fire Risk Radar",
        "category": "Due-Diligence & Compliance", 
        "description": "Maps FEMA flood, wildfire, and climate-risk layers for a property and translates them into guidance.",
        "credits_cost": 3,
        "inputs": ["property_address"],
        "outputs": ["risk_map", "premium_impact_estimate"]
    },
    "hoa_decoder": {
        "id": "hoa_decoder",
        "name": "HOA Decoder",
        "category": "Due-Diligence & Compliance",
        "description": "Summarizes 50-page HOA docs into readable rules, fees, pet restrictions, and rental caps.",
        "credits_cost": 2,
        "inputs": ["hoa_documents"],
        "outputs": ["summary", "deal_breaker_highlights"]
    },
    "zoning_whisperer": {
        "id": "zoning_whisperer",
        "name": "Zoning Whisperer",
        "category": "Due-Diligence & Compliance",
        "description": "Translates local zoning code into plain English and shows what can/can't be built or ADU-ed.",
        "credits_cost": 2,
        "inputs": ["property_address"],
        "outputs": ["buildout_matrix", "permit_path"]
    },
    "renovation_rulebook": {
        "id": "renovation_rulebook",
        "name": "Renovation-Rulebook Retriever", 
        "category": "Due-Diligence & Compliance",
        "description": "Pulls city/county renovation rules, flags common pitfalls, and links to permit forms.",
        "credits_cost": 2,
        "inputs": ["property_address"],
        "outputs": ["compliance_checklist", "permit_links"]
    },
    
    # Market Intel & Strategy
    "neighborhood_insider": {
        "id": "neighborhood_insider",
        "name": "Neighborhood Insider",
        "category": "Market Intel & Strategy",
        "description": "Generates a shareable local intel sheet—schools, commute times, walk scores, new developments, vibe.",
        "credits_cost": 3,
        "inputs": ["address", "lat_long"],
        "outputs": ["intel_pdf", "commentary_hooks"]
    },
    "comp_cruncher_cma": {
        "id": "comp_cruncher_cma",
        "name": "Comp-Cruncher CMA", 
        "category": "Market Intel & Strategy",
        "description": "Builds a plain-English comparative-market analysis and price-position recommendation.",
        "credits_cost": 4,
        "inputs": ["subject_property", "mls_data"],
        "outputs": ["price_graph", "list_price_range", "talking_points"]
    },
    "investor_math_buddy": {
        "id": "investor_math_buddy",
        "name": "Investor Math Buddy",
        "category": "Market Intel & Strategy",
        "description": "Calculates cap rate, cash-on-cash, IRR, break-even rents from a quickfill worksheet.",
        "credits_cost": 2,
        "inputs": ["purchase_price", "rents", "expenses", "loan_terms"],
        "outputs": ["analysis_table", "pros_cons_narrative"]
    },
    "renovation_roi_estimator": {
        "id": "renovation_roi_estimator",
        "name": "Renovation ROI Estimator",
        "category": "Market Intel & Strategy",
        "description": "Predicts value-add lift for common upgrades and ranks them by expected return and days-on-market impact.",
        "credits_cost": 3,
        "inputs": ["property_type", "comps", "upgrade_list"],
        "outputs": ["roi_chart", "project_recommendations"]
    },
    "school_scope_analyzer": {
        "id": "school_scope_analyzer", 
        "name": "SchoolScope Analyzer",
        "category": "Market Intel & Strategy",
        "description": "Generates a parent-friendly breakdown of public/private schools within X miles.",
        "credits_cost": 2,
        "inputs": ["location", "school_radius"],
        "outputs": ["school_cheatsheet", "talking_points"]
    },
    "farm_area_crystal_ball": {
        "id": "farm_area_crystal_ball",
        "name": "Farm-Area Crystal Ball",
        "category": "Market Intel & Strategy", 
        "description": "Tracks demographic shifts, new permits, and price trends in a farming area; spits out a report.",
        "credits_cost": 4,
        "inputs": ["farming_area"],
        "outputs": ["trend_charts", "opportunity_score"]
    },
    "investor_exit_planner": {
        "id": "investor_exit_planner",
        "name": "Investor Exit Planner",
        "category": "Market Intel & Strategy",
        "description": "Generates multiple sell/hold/refi exit strategies for small landlords based on forecasts.",
        "credits_cost": 3,
        "inputs": ["rent_roll", "market_forecasts"],
        "outputs": ["roi_scenarios"]
    },
    "fair_rent_finder": {
        "id": "fair_rent_finder",
        "name": "Fair-Rent Finder",
        "category": "Market Intel & Strategy",
        "description": "Benchmarks local rental rates, vacancy, and absorption for investor clients or STR pricing.",
        "credits_cost": 2,
        "inputs": ["location", "property_type"],
        "outputs": ["rent_matrix", "sweet_spot_range"]
    },
    
    # Process & Productivity
    "open_house_orchestrator": {
        "id": "open_house_orchestrator",
        "name": "Open-House Orchestrator", 
        "category": "Process & Productivity",
        "description": "Plans the event end-to-end: invite copy, SMS reminders, sign-in QR form, post-event drip emails.",
        "credits_cost": 3,
        "inputs": ["date_time", "buyer_persona", "followup_cadence"],
        "outputs": ["calendar_ics", "sms_templates", "email_sequence"]
    },
    "lead_qualifier_lite": {
        "id": "lead_qualifier_lite",
        "name": "Lead Qualifier Lite",
        "category": "Process & Productivity",
        "description": "Interviews inbound leads via chat/text, scores motivation & financing, and pushes hot leads to the CRM.",
        "credits_cost": 2,
        "inputs": ["lead_inquiry"],
        "outputs": ["qualification_summary", "urgency_score", "next_action"]
    },
    "relocation_concierge": {
        "id": "relocation_concierge",
        "name": "Relocation Concierge",
        "category": "Process & Productivity",
        "description": "Crafts a turnkey welcome packet with utilities, DMV, healthcare, hotspots, and kid-friendly recs.",
        "credits_cost": 2,
        "inputs": ["destination_zip", "family_profile"],
        "outputs": ["welcome_packet", "resource_links"]
    },
    "energy_saver_scorecard": {
        "id": "energy_saver_scorecard",
        "name": "Energy-Saver Scorecard",
        "category": "Process & Productivity",
        "description": "Estimates utility costs & carbon footprint vs. comps; recommends top 3 ROI-positive efficiency upgrades.",
        "credits_cost": 2,
        "inputs": ["property_details", "comparable_properties"],
        "outputs": ["scorecard_graphic", "rebate_links"]
    },
    "expired_listing_resurrector": {
        "id": "expired_listing_resurrector",
        "name": "Expired-Listing Resurrector",
        "category": "Process & Productivity",
        "description": "Autopsies an expired MLS entry, diagnoses why it didn't sell, and drafts a relaunch game-plan.",
        "credits_cost": 3,
        "inputs": ["expired_mls_data"],
        "outputs": ["fix_list", "relaunch_copy"]
    },
    "voice_note_summarizer": {
        "id": "voice_note_summarizer",
        "name": "Voice-Note Summarizer", 
        "category": "Process & Productivity",
        "description": "Converts messy on-the-road voice memos into clean client updates, task lists, or CRM notes.",
        "credits_cost": 1,
        "inputs": ["voice_recording"],
        "outputs": ["structured_text", "follow_ups"]
    },
    "open_house_debrief_bot": {
        "id": "open_house_debrief_bot",
        "name": "Open-House Debrief Bot",
        "category": "Process & Productivity",
        "description": "After the event, digests sign-in data & attendee feedback, then drafts personalized follow-up emails ranked by lead quality.",
        "credits_cost": 2,
        "inputs": ["signin_data", "feedback"],
        "outputs": ["lead_scoresheet", "email_merge_file"]
    },
    "sellers_stress_buster": {
        "id": "sellers_stress_buster",
        "name": "Sellers' Stress-Buster",
        "category": "Process & Productivity",
        "description": "Creates a personalized selling prep calendar that spaces out decluttering, minor fixes, showings, and move-out tasks.",
        "credits_cost": 2,
        "inputs": ["property_details", "sale_timeline"],
        "outputs": ["ical_file", "printable_checklist"]
    }
}

# Authentication functions complete

# Admin endpoints
@api_router.post("/admin/login", response_model=Token)
async def admin_login(user_data: UserLogin):
    """Admin login"""
    try:
        # Find admin
        admin = await db.admin_users.find_one({"email": user_data.email})
        if not admin or not verify_password(user_data.password, admin["hashed_password"]):
            raise HTTPException(status_code=401, detail="Incorrect email or password")
        
        # Create access token
        access_token = create_access_token(data={"sub": admin["id"]})
        
        # Convert to user-like response for token compatibility
        admin_as_user = User(
            id=admin["id"],
            email=admin["email"],
            full_name=admin["full_name"],
            is_active=True,
            credits=0,  # Admins don't need credits
            subscription_status="admin",
            referral_code="ADMIN",
            created_at=admin["created_at"]
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user=admin_as_user
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@api_router.get("/admin/users")
async def get_all_users(
    skip: int = 0, 
    limit: int = 50,
    current_admin: AdminUser = Depends(get_current_admin_user)
):
    """Get all users with pagination"""
    try:
        users = []
        async for user in db.users.find().skip(skip).limit(limit).sort("created_at", -1):
            # Remove sensitive data
            user_data = {k: v for k, v in user.items() if k != 'hashed_password'}
            if '_id' in user_data:
                user_data['_id'] = str(user_data['_id'])
            users.append(user_data)
        
        total_users = await db.users.count_documents({})
        
        return {
            "users": users,
            "total": total_users,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error retrieving users: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")

@api_router.put("/admin/users/{user_id}/credits")
async def update_user_credits(
    user_id: str,
    credits: int,
    current_admin: AdminUser = Depends(get_current_admin_user)
):
    """Update user's credit balance"""
    try:
        result = await db.users.update_one(
            {"id": user_id},
            {"$set": {"credits": credits}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": f"User credits updated to {credits}"}
    except Exception as e:
        logger.error(f"Error updating user credits: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update credits")

@api_router.get("/admin/tool-rates")
async def get_tool_rates(current_admin: AdminUser = Depends(get_current_admin_user)):
    """Get all tool rates"""
    try:
        rates = []
        async for rate in db.tool_rates.find():
            if '_id' in rate:
                rate['_id'] = str(rate['_id'])
            rates.append(rate)
        
        # Add default rates if not in database
        if not rates:
            default_rates = [
                {"tool_name": "interior_design", "credits_per_use": 5, "description": "AI Interior Design Generation"},
                {"tool_name": "gpt_concept", "credits_per_use": 1, "description": "GPT Concept Generation"}
            ]
            await db.tool_rates.insert_many(default_rates)
            return {"rates": default_rates}
        
        return {"rates": rates}
    except Exception as e:
        logger.error(f"Error retrieving tool rates: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tool rates")

@api_router.put("/admin/tool-rates/{tool_name}")
async def update_tool_rate(
    tool_name: str,
    credits_per_use: int,
    description: str = "",
    current_admin: AdminUser = Depends(get_current_admin_user)
):
    """Update tool credit rate"""
    try:
        result = await db.tool_rates.update_one(
            {"tool_name": tool_name},
            {"$set": {"credits_per_use": credits_per_use, "description": description}},
            upsert=True
        )
        
        return {"message": f"Tool rate updated: {tool_name} = {credits_per_use} credits"}
    except Exception as e:
        logger.error(f"Error updating tool rate: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update tool rate")

@api_router.get("/admin/analytics")
async def get_analytics(current_admin: AdminUser = Depends(get_current_admin_user)):
    """Get platform analytics"""
    try:
        # User statistics
        total_users = await db.users.count_documents({})
        active_subscribers = await db.users.count_documents({"subscription_status": "active"})
        free_users = await db.users.count_documents({"subscription_status": "free"})
        
        # Usage statistics
        total_designs = await db.interior_designs.count_documents({})
        designs_today = await db.interior_designs.count_documents({
            "created_at": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)}
        })
        
        # Credit statistics
        pipeline = [
            {"$group": {"_id": None, "total_credits": {"$sum": "$credits"}}}
        ]
        credit_result = await db.users.aggregate(pipeline).to_list(1)
        total_credits_in_circulation = credit_result[0]["total_credits"] if credit_result else 0
        
        return {
            "users": {
                "total": total_users,
                "active_subscribers": active_subscribers,
                "free_users": free_users
            },
            "usage": {
                "total_designs": total_designs,
                "designs_today": designs_today
            },
            "credits": {
                "total_in_circulation": total_credits_in_circulation
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")

class AdminCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    setup_key: str

# Development/Setup endpoints (remove in production)
@api_router.post("/setup/create-admin")
async def create_admin_user_endpoint(admin_data: AdminCreate):
    """Create admin user (development only)"""
    if admin_data.setup_key != "SETUP_ADMIN_2024":
        raise HTTPException(status_code=403, detail="Invalid setup key")
    
    try:
        # Check if admin already exists
        existing_admin = await db.admin_users.find_one({"email": admin_data.email})
        if existing_admin:
            raise HTTPException(status_code=400, detail="Admin already exists")
        
        # Hash password
        hashed_password = get_password_hash(admin_data.password)
        
        # Create admin user
        admin_user = {
            "id": str(uuid.uuid4()),
            "email": admin_data.email,
            "full_name": admin_data.full_name,
            "hashed_password": hashed_password,
            "role": "admin",
            "created_at": datetime.utcnow()
        }
        
        # Insert admin user
        await db.admin_users.insert_one(admin_user)
        
        return {"message": f"Admin user created: {admin_data.email}"}
        
    except Exception as e:
        logger.error(f"Error creating admin: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create admin")

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
    room_type: str = Form("living_room"),
    designer: str = Form("alessia_duval"), 
    color_scheme: str = Form("glacial_muse"),
    current_user: User = Depends(get_current_user_enhanced)
):
    """Process an interior image with custom design preferences (requires authentication)"""
    try:
        # Check credits and deduct
        credits_needed = await get_tool_rate('interior_design')
        if not await deduct_credits(current_user.id, credits_needed):
            raise HTTPException(status_code=402, detail=f"Insufficient credits. Need {credits_needed} credits.")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            # Refund credits if file validation fails
            await db.users.update_one(
                {"id": current_user.id},
                {"$inc": {"credits": credits_needed}}
            )
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
        
        # Create database record with preferences and user ID
        design_request = InteriorDesignRequest(
            original_filename=file.filename,
            status="queued",
            room_type=room_type,
            designer=designer,
            color_scheme=color_scheme,
            generated_prompt=generated_prompt
        )
        
        # Add user ID to the database record
        design_dict = design_request.dict()
        design_dict["user_id"] = current_user.id
        design_dict["credits_used"] = credits_needed
        
        await db.interior_designs.insert_one(design_dict)
        
        # Return immediately with queue status - process asynchronously
        import asyncio
        asyncio.create_task(process_image_async(design_request.id, temp_file_path, generated_prompt, file.filename, room_type, designer, color_scheme))
        
        return {
            "id": design_request.id,
            "status": "queued",
            "message": "Your design request has been queued for processing",
            "original_filename": file.filename,
            "room_type": room_type,
            "designer": designer,
            "color_scheme": color_scheme,
            "generated_prompt": generated_prompt,
            "credits_used": credits_needed,
            "remaining_credits": current_user.credits - credits_needed
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        # Refund credits on error
        try:
            credits_needed = await get_tool_rate('interior_design')
            await db.users.update_one(
                {"id": current_user.id},
                {"$inc": {"credits": credits_needed}}
            )
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

async def download_and_store_image(image_url: str, request_id: str) -> str:
    """Download image from URL and store it permanently"""
    try:
        # Generate filename
        filename = f"{request_id}.jpg"
        file_path = PROCESSED_IMAGES_DIR / filename
        
        # Download image
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    async with aiofiles.open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    
                    # Return the local URL path
                    return f"/api/images/{filename}"
                else:
                    logger.error(f"Failed to download image: {response.status}")
                    return image_url  # Return original URL as fallback
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        return image_url  # Return original URL as fallback

async def process_image_async(request_id: str, temp_file_path, generated_prompt: str, filename: str, room_type: str, designer: str, color_scheme: str):
    """Process image asynchronously to allow queue functionality"""
    try:
        # Update status to processing
        await db.interior_designs.update_one(
            {"id": request_id},
            {"$set": {"status": "processing"}}
        )
        
        # TEMPORARY: Switch to working Replicate model while RunPod is being fixed
        import replicate
        
        with open(temp_file_path, "rb") as image_file:
            # Use a working interior design model with custom prompt and higher resolution
            output = replicate.run(
                "adirik/interior-design:76604baddc85b1b4616e1c6475eca080da339c8875bd4996705440484a6eac38",
                input={
                    "image": image_file,
                    "prompt": generated_prompt,
                    "width": 1024,  # Higher resolution
                    "height": 1024,  # Higher resolution
                    "num_inference_steps": 50,  # Better quality
                    "guidance_scale": 7.5,  # Better prompt adherence
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
            
            # Download and store the image permanently
            logger.info(f"Downloading and storing image from: {processed_url}")
            local_image_url = await download_and_store_image(processed_url, request_id)
            logger.info(f"Image stored locally at: {local_image_url}")
            
            # Apply watermark if user has branding configured
            watermarked_url = None
            try:
                from watermark_utils import ensure_image_watermarked
                
                # Get the local file path
                local_filename = local_image_url.split("/")[-1]
                image_path = PROCESSED_IMAGES_DIR / local_filename
                
                # Get user ID from the design record
                design_record = await db.interior_designs.find_one({"id": request_id})
                if design_record and design_record.get("user_id"):
                    watermarked_path = await ensure_image_watermarked(
                        str(image_path), 
                        design_record["user_id"], 
                        db
                    )
                    
                    if watermarked_path != str(image_path):
                        watermarked_filename = Path(watermarked_path).name
                        watermarked_url = f"/api/images/{watermarked_filename}"
                        logger.info(f"Watermark applied to interior design: {watermarked_url}")
                        
            except Exception as e:
                logger.error(f"Watermarking failed for design {request_id}: {str(e)}")
                # Continue without watermark
            
            # Update database with success
            update_data = {
                "status": "completed",
                "processed_image_url": local_image_url,  # Use local URL
                "original_replicate_url": processed_url,  # Keep original URL for reference
                "completed_at": datetime.utcnow()
            }
            
            # Add watermarked URL if available
            if watermarked_url:
                update_data["watermarked_image_url"] = watermarked_url
            
            await db.interior_designs.update_one(
                {"id": request_id},
                {"$set": update_data}
            )
            
    except Exception as e:
        logger.error(f"Error processing image async: {str(e)}")
        # Update database with error
        await db.interior_designs.update_one(
            {"id": request_id},
            {"$set": {
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.utcnow()
            }}
        )
    finally:
        # Clean up temp file
        try:
            temp_file_path.unlink()
        except:
            pass

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
async def get_interior_design_history(current_user: User = Depends(get_current_user_enhanced)):
    """Get user's interior design processing history"""
    try:
        designs_list = []
        async for design in db.interior_designs.find({"user_id": current_user.id}).sort("created_at", -1).limit(20):
            # Convert ObjectId to string for JSON serialization
            if '_id' in design:
                design['_id'] = str(design['_id'])
            designs_list.append(design)
        return {"designs": designs_list}
    except Exception as e:
        logger.error(f"Error retrieving interior design history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@api_router.get("/interior-design/queue")
async def get_queue_status():
    """Get current queue status"""
    try:
        # Count designs by status
        queued_count = await db.interior_designs.count_documents({"status": "queued"})
        processing_count = await db.interior_designs.count_documents({"status": "processing"})
        
        # Get recent queue items and convert ObjectId to string
        queued_items = []
        async for item in db.interior_designs.find({"status": "queued"}).sort("upload_timestamp", 1).limit(10):
            if '_id' in item:
                item['_id'] = str(item['_id'])
            queued_items.append(item)
        
        processing_items = []
        async for item in db.interior_designs.find({"status": "processing"}).sort("upload_timestamp", 1).limit(5):
            if '_id' in item:
                item['_id'] = str(item['_id'])
            processing_items.append(item)
        
        return {
            "queue_status": {
                "queued": queued_count,
                "processing": processing_count
            },
            "queued_items": queued_items,
            "processing_items": processing_items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/interior-design/delete/{design_id}")
async def delete_design(design_id: str):
    """Delete a design and its associated image file"""
    try:
        design = await db.interior_designs.find_one({"id": design_id})
        if not design:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Delete the image file if it exists locally
        local_url = design.get("processed_image_url", "")
        if local_url.startswith("/api/images/"):
            filename = local_url.split("/")[-1]
            file_path = PROCESSED_IMAGES_DIR / filename
            
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Deleted image file: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to delete image file {filename}: {str(e)}")
        
        # Delete the database record
        result = await db.interior_designs.delete_one({"id": design_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Design not found in database")
        
        return {
            "success": True,
            "message": "Design deleted successfully",
            "deleted_id": design_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting design: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/images/{filename}")
async def serve_processed_image(filename: str):
    """Serve processed images"""
    file_path = PROCESSED_IMAGES_DIR / filename
    if file_path.exists() and file_path.is_file():
        return FileResponse(
            path=file_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=31536000"}  # 1 year cache
        )
    raise HTTPException(status_code=404, detail="Image not found")

@api_router.get("/interior-design/download/{design_id}")
async def download_design_image(design_id: str):
    """Download processed design image"""
    try:
        design = await db.interior_designs.find_one({"id": design_id})
        if not design:
            raise HTTPException(status_code=404, detail="Design not found")
        
        if design["status"] != "completed" or not design.get("processed_image_url"):
            raise HTTPException(status_code=400, detail="Design not completed or image not available")
        
        # Check if we have a local image file
        local_url = design.get("processed_image_url", "")
        if local_url.startswith("/api/images/"):
            filename = local_url.split("/")[-1]
            file_path = PROCESSED_IMAGES_DIR / filename
            
            if file_path.exists():
                return FileResponse(
                    path=file_path,
                    media_type="image/jpeg",
                    filename=f"ai_design_{design_id}.jpg",
                    headers={"Content-Disposition": "attachment"}
                )
        
        # Fallback to original URL if local file not found
        original_url = design.get("original_replicate_url")
        if original_url:
            return {"download_url": original_url, "filename": f"ai_design_{design_id}.jpg"}
        
        raise HTTPException(status_code=404, detail="Image file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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