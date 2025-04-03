# inside predict.py
import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from peft import get_peft_model, LoraConfig

LORA_PATH = "pytorch_lora_weights.bin"

# Force Replicate to allow upload by checking if file exists
assert os.path.exists(LORA_PATH), f"Missing required file: {LORA_PATH}"

# Load base model
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.unet = get_peft_model(pipe.unet, LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION"
))
pipe.unet.load_state_dict(torch.load(LORA_PATH), strict=False)
