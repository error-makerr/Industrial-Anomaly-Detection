# Run this in Jupyter to CREATE the script
script = '''
import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
import json

# Config
IMG_DIR    = "/home/sufi/lora_training/datasets/contamination/images"
OUTPUT_DIR = "/home/sufi/lora_training/weights/contamination"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
EPOCHS     = 100
LR         = 1e-4
BATCH_SIZE = 2
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model components...")
tokenizer   = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
text_encoder= CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder").cuda()
vae         = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae").cuda()
unet        = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet").cuda()
scheduler   = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

# Freeze everything except LoRA
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# Add LoRA to UNet attention layers
lora_attn_procs = {}
for name in unet.attn_processors.keys():
    cross = name.endswith("attn2.processor")
    if cross:
        hidden = unet.config.cross_attention_dim
    else:
        hidden = unet.config.attention_head_dim
        if isinstance(hidden, list):
            layer_name = name.split(".processor")[0]
            parts = layer_name.split(".")
            block_id = int(parts[1]) if parts[1].isdigit() else 0
            hidden = unet.config.attention_head_dim[block_id]
    lora_attn_procs[name] = LoRAAttnProcessor(
        hidden_size=hidden,
        cross_attention_dim=unet.config.cross_attention_dim if cross else None,
        rank=4
    ).cuda()
unet.set_attn_processor(lora_attn_procs)
lora_layers = AttnProcsLayers(unet.attn_processors)

optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=LR)

# Dataset
class DefectDataset(Dataset):
    def __init__(self, img_dir):
        self.imgs = sorted([
            f for f in Path(img_dir).glob("*.png")
            if "metadata" not in str(f)
        ])
        self.txts = {p: p.with_suffix(".txt") for p in self.imgs}
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5]),
        ])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = self.transform(Image.open(self.imgs[i]).convert("RGB"))
        cap = open(self.txts[self.imgs[i]]).read().strip()
        return img, cap

dataset = DefectDataset(IMG_DIR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f"Dataset: {len(dataset)} images")

scaler = torch.cuda.amp.GradScaler()

print("Starting LoRA training...")
for epoch in range(EPOCHS):
    total_loss = 0
    for imgs, captions in loader:
        imgs = imgs.cuda().half()
        
        # Encode images to latents
        with torch.no_grad():
            latents = vae.encode(imgs).latent_dist.sample() * 0.18215
        
        # Add noise
        noise     = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                  (latents.shape[0],), device="cuda").long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        # Encode text
        with torch.no_grad():
            tokens = tokenizer(list(captions), padding="max_length",
                               max_length=77, truncation=True,
                               return_tensors="pt").input_ids.cuda()
            text_emb = text_encoder(tokens)[0]
        
        # Predict noise
        with torch.cuda.amp.autocast():
            pred  = unet(noisy_latents, timesteps, text_emb).sample
            loss  = F.mse_loss(pred.float(), noise.float())
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    
    avg = total_loss / len(loader)
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}]  loss={avg:.4f}")

# Save LoRA weights
lora_layers.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA saved to {OUTPUT_DIR}")
'''

with open("/home/sufi/lora_training/train_lora.py", "w") as f:
    f.write(script)

print("✅ Script written to /home/sufi/lora_training/train_lora.py")
print("\nNow open WSL terminal and run:")
print("   conda activate organized")
print("   python /home/sufi/lora_training/train_lora.py")