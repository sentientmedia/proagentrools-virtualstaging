"""
Watermarking utilities for ProAgentTools
Applies agent logos to all images (interior designs, AI outputs, uploads)
"""

import asyncio
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging
from PIL import Image, ImageEnhance
import aiofiles
import io

logger = logging.getLogger(__name__)

class WatermarkProcessor:
    """Handles watermarking of images with agent branding"""
    
    def __init__(self):
        self.supported_formats = {'JPEG', 'PNG', 'JPG', 'WEBP'}
        
    async def apply_watermark(
        self,
        image_path: str,
        logo_path: str,
        position: str = "bottom-right",
        opacity: float = 0.7,
        logo_size_ratio: float = 0.15
    ) -> str:
        """
        Apply watermark to an image
        
        Args:
            image_path: Path to the original image
            logo_path: Path to the logo/watermark
            position: Watermark position (bottom-right, bottom-left, etc.)
            opacity: Logo opacity (0.0 to 1.0)
            logo_size_ratio: Logo size as ratio of image width
            
        Returns:
            Path to the watermarked image
        """
        try:
            # Load the main image
            with Image.open(image_path) as main_img:
                main_img = main_img.convert('RGBA')
                
                # Load and prepare the logo
                with Image.open(logo_path) as logo_img:
                    logo_img = logo_img.convert('RGBA')
                    
                    # Resize logo based on main image size
                    logo_width = int(main_img.width * logo_size_ratio)
                    logo_height = int(logo_img.height * (logo_width / logo_img.width))
                    logo_resized = logo_img.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
                    
                    # Apply opacity
                    if opacity < 1.0:
                        logo_resized = self._apply_opacity(logo_resized, opacity)
                    
                    # Calculate position
                    pos_x, pos_y = self._calculate_position(
                        main_img.size, logo_resized.size, position
                    )
                    
                    # Create a transparent overlay
                    overlay = Image.new('RGBA', main_img.size, (0, 0, 0, 0))
                    overlay.paste(logo_resized, (pos_x, pos_y), logo_resized)
                    
                    # Composite the images
                    watermarked = Image.alpha_composite(main_img, overlay)
                    
                    # Convert back to RGB for JPEG saving
                    if watermarked.mode == 'RGBA':
                        rgb_img = Image.new('RGB', watermarked.size, (255, 255, 255))
                        rgb_img.paste(watermarked, mask=watermarked.split()[-1])
                        watermarked = rgb_img
                    
                    # Generate output path
                    output_path = self._generate_watermarked_path(image_path)
                    
                    # Save the watermarked image
                    watermarked.save(output_path, 'JPEG', quality=95)
                    
                    logger.info(f"Watermark applied: {output_path}")
                    return output_path
                    
        except Exception as e:
            logger.error(f"Watermark application failed: {str(e)}")
            raise
    
    def _apply_opacity(self, image: Image.Image, opacity: float) -> Image.Image:
        """Apply opacity to an image"""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get the alpha channel
        alpha = image.split()[-1]
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
        
        # Replace the alpha channel
        image.putalpha(alpha)
        return image
    
    def _calculate_position(
        self, 
        main_size: Tuple[int, int], 
        logo_size: Tuple[int, int], 
        position: str
    ) -> Tuple[int, int]:
        """Calculate logo position based on position string"""
        
        main_width, main_height = main_size
        logo_width, logo_height = logo_size
        
        # Padding from edges
        padding = 20
        
        position_map = {
            "top-left": (padding, padding),
            "top-right": (main_width - logo_width - padding, padding),
            "bottom-left": (padding, main_height - logo_height - padding),
            "bottom-right": (main_width - logo_width - padding, main_height - logo_height - padding),
            "center": ((main_width - logo_width) // 2, (main_height - logo_height) // 2)
        }
        
        return position_map.get(position, position_map["bottom-right"])
    
    def _generate_watermarked_path(self, original_path: str) -> str:
        """Generate path for watermarked image"""
        path = Path(original_path)
        stem = path.stem
        suffix = path.suffix
        
        # Add watermarked suffix
        watermarked_filename = f"{stem}_watermarked{suffix}"
        return str(path.parent / watermarked_filename)
    
    async def batch_watermark_images(
        self,
        image_paths: list,
        logo_path: str,
        position: str = "bottom-right",
        opacity: float = 0.7
    ) -> list:
        """Apply watermarks to multiple images concurrently"""
        
        tasks = []
        for image_path in image_paths:
            task = asyncio.create_task(
                self.apply_watermark(image_path, logo_path, position, opacity)
            )
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            watermarked_paths = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to watermark {image_paths[i]}: {result}")
                    watermarked_paths.append(None)
                else:
                    watermarked_paths.append(result)
            
            return watermarked_paths
            
        except Exception as e:
            logger.error(f"Batch watermarking failed: {str(e)}")
            raise

class InteriorDesignWatermarker:
    """Specialized watermarking for interior design images"""
    
    def __init__(self):
        self.processor = WatermarkProcessor()
    
    async def watermark_interior_design(
        self, 
        design_id: str, 
        user_id: str,
        db_client
    ) -> Optional[str]:
        """Watermark an interior design image"""
        
        try:
            # Get the design from database
            design = await db_client.interior_designs.find_one({"design_id": design_id})
            if not design:
                logger.error(f"Design not found: {design_id}")
                return None
            
            # Check if design belongs to user
            if design.get("user_id") != user_id:
                logger.error(f"Design access denied: {design_id}")
                return None
            
            # Get user's branding settings
            branding = await db_client.agent_branding.find_one({"user_id": user_id})
            if not branding or not branding.get("logo_url"):
                logger.info(f"No logo found for user: {user_id}")
                return None
            
            # Get image path
            image_path = Path("storage/processed_images") / design.get("filename", "")
            if not image_path.exists():
                logger.error(f"Design image not found: {image_path}")
                return None
            
            # Get logo path
            logo_filename = branding["logo_url"].split("/")[-1]
            logo_path = Path("storage/branding") / logo_filename
            if not logo_path.exists():
                logger.error(f"Logo not found: {logo_path}")
                return None
            
            # Apply watermark
            watermarked_path = await self.processor.apply_watermark(
                str(image_path),
                str(logo_path),
                branding.get("watermark_position", "bottom-right"),
                branding.get("watermark_opacity", 0.7)
            )
            
            # Update database with watermarked version
            watermarked_filename = Path(watermarked_path).name
            await db_client.interior_designs.update_one(
                {"design_id": design_id},
                {
                    "$set": {
                        "watermarked_filename": watermarked_filename,
                        "watermarked_url": f"/api/images/{watermarked_filename}",
                        "watermarked_at": asyncio.get_event_loop().time()
                    }
                }
            )
            
            return watermarked_path
            
        except Exception as e:
            logger.error(f"Interior design watermarking failed: {str(e)}")
            return None

# Utility functions
async def ensure_image_watermarked(
    image_path: str,
    user_id: str,
    db_client,
    force_refresh: bool = False
) -> Optional[str]:
    """Ensure an image is watermarked with user's branding"""
    
    try:
        # Check if watermarked version already exists
        watermarked_path = WatermarkProcessor()._generate_watermarked_path(image_path)
        
        if Path(watermarked_path).exists() and not force_refresh:
            return watermarked_path
        
        # Get user's branding
        branding = await db_client.agent_branding.find_one({"user_id": user_id})
        if not branding or not branding.get("logo_url"):
            return image_path  # Return original if no logo
        
        # Get logo path
        logo_filename = branding["logo_url"].split("/")[-1]
        logo_path = Path("storage/branding") / logo_filename
        
        if not logo_path.exists():
            return image_path  # Return original if logo missing
        
        # Apply watermark
        processor = WatermarkProcessor()
        watermarked_path = await processor.apply_watermark(
            image_path,
            str(logo_path),
            branding.get("watermark_position", "bottom-right"),
            branding.get("watermark_opacity", 0.7)
        )
        
        return watermarked_path
        
    except Exception as e:
        logger.error(f"Image watermarking failed: {str(e)}")
        return image_path  # Return original on error