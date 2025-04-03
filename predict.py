from cog import BasePredictor, Input, Path
from typing import Optional
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDPMScheduler, UNet2DConditionModel
from safetensors.torch import load_file

class Predictor(BasePredictor):
    def setup(self):
        # Load base SD 1.5 pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")

        # 🔁 Replace UNet with your fine-tuned model
        unet = UNet2DConditionModel.from_pretrained(
            "sentientmedia/patvs",  # your Hugging Face repo
            subfolder="",  # root folder (where .safetensors is)
            torch_dtype=torch.float16
        )
        self.pipe.unet = unet

    def predict(
        self,
        prompt: str = Input(description="Prompt to generate image"),
        init_image: Path = Input(description="Input image (before room)"),
        strength: float = Input(default=0.75, description="Prompt strength"),
        guidance_scale: float = Input(default=7.5, description="Prompt guidance scale")
    ) -> Path:
        image = Image.open(init_image).convert("RGB")
        output = self.pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale
        )
        output_path = "/tmp/generated.png"
        output.images[0].save(output_path)
        return Path(output_path)
