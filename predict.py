from cog import BasePredictor, Input, Path
from typing import Optional
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDPMScheduler, UNet2DConditionModel

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")

        # Load your trained UNet
        self.pipe.unet = UNet2DConditionModel.from_pretrained(
            "sentientmedia/patvs",  # Your trained model on HF
            torch_dtype=torch.float16
        )

    def predict(
        self,
        prompt: str = Input(description="Prompt describing the 'after' version of this room"),
        image: Path = Input(description="Input image — the 'before' room photo"),
        strength: float = Input(default=0.75, ge=0.0, le=1.0, description="How much to transform the image (0 = minimal, 1 = strong)"),
        guidance: float = Input(default=7.5, ge=1.0, le=20.0, description="How much the prompt guides the output")
    ) -> Path:
        init_image = Image.open(image).convert("RGB")
        result = self.pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance
        )
        output_path = "/tmp/after.png"
        result.images[0].save(output_path)
        return Path(output_path)

