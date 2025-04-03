from cog import BasePredictor, Input, Path
from typing import Optional
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDPMScheduler, UNet2DConditionModel
from safetensors.torch import load_file

class Predictor(BasePredictor):
    def setup(self):
        # Load base pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")

        # Replace UNet with your trained model from Hugging Face
        unet = UNet2DConditionModel.from_pretrained(
            "sentientmedia/patvs",           # 🔗 Your Hugging Face model
            subfolder="",                    # Files are in the root
            torch_dtype=torch.float16
        )
        self.pipe.unet = unet

    def predict(
        self,
        prompt: str = Input(description="Prompt to describe the 'after' design (e.g. 'luxury modern living room after renovation')"),
        init_image: Path = Input(description="Input image (the 'before' room)"),
        strength: float = Input(default=0.75, ge=0.0, le=1.0, description="How much to transform the input image (0 = minimal, 1 = full change)"),
        guidance_scale: float = Input(default=7.5, ge=1.0, le=20.0, description="Prompt strength (higher = more influence from prompt)")
    ) -> Path:
        image = Image.open(init_image).convert("RGB")
        result = self.pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale
        )
        output_path = "/tmp/generated.png"
        result.images[0].save(output_path)
        return Path(output_path)
