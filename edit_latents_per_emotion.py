import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from models.stylegan2.model import Generator
from utils.text_direction import get_direction
import numpy as np

# ========== CONFIG ==========
device = 'cpu'
latent_dir = 'inversions'
output_dir = 'edited_per_emotion'
g_path = 'pretrained_models/stylegan2-ffhq-config-f.pt'
os.makedirs(output_dir, exist_ok=True)
save_comparison = True  # Set to False if you don’t want original+edited output

# ========== EMOTION MAP ==========
emotion_flip = {
    "happy": "an angry face",
    "angry": "a smiling face",
    "sad": "a smiling face",
    "fear": "a calm face",
    "disgust": "a pleasant face",
    "neutral": "a surprised face",
    "surprise": "a neutral face"
}

contrastive_pairs = {
    "happy": ("a smiling face", "an angry face"),
    "angry": ("an angry face", "a smiling face"),
    "sad": ("a sad face", "a smiling face"),
    "fear": ("a fearful face", "a calm face"),
    "disgust": ("a disgusted face", "a pleasant face"),
    "neutral": ("a neutral face", "a surprised face"),
    "surprise": ("a surprised face", "a neutral face")
}

# ========== HELPER FUNCTION ==========
def apply_identity_preserving_edit(latent, direction, layers=[4,5,6,7], strength=1.5, blend_ratio=0.3):
    direction = direction / direction.norm()
    edited = latent.clone()
    for i in layers:
        raw_edit = latent[:, i, :] + strength * direction
        edited[:, i, :] = (1 - blend_ratio) * latent[:, i, :] + blend_ratio * raw_edit
    return torch.clamp(edited, -2.0, 2.0)

# ========== LOAD GENERATOR ==========
generator = Generator(1024, 512, 8)
ckpt = torch.load(g_path, map_location=device)
generator.load_state_dict(ckpt["g_ema"], strict=False)
generator.eval().to(device)

# ========== MAIN LOOP ==========
df = pd.read_csv("emotions.csv")
direction_cache = {}
success_count = 0
skipped = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    fname = row["image"]
    emotion = str(row["emotion"]).strip().lower()

    if emotion not in emotion_flip:
        print(f"⚠️ Unknown emotion '{emotion}' for {fname}")
        skipped.append(fname)
        continue

    latent_path = os.path.join(latent_dir, fname.replace(".jpg", ".pt").replace(".png", ".pt"))
    if not os.path.exists(latent_path):
        print(f"❌ Missing latent for {fname}")
        skipped.append(fname)
        continue

    try:
        latent = torch.load(latent_path, map_location=device)
    except Exception as e:
        print(f"❌ Failed to load {latent_path}: {e}")
        skipped.append(fname)
        continue

    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    if latent.shape != (1, 18, 512):
        print(f"❌ Bad shape for {fname}: {latent.shape}")
        skipped.append(fname)
        continue

    src_prompt, tgt_prompt = contrastive_pairs[emotion]
    key = f"{src_prompt}->{tgt_prompt}"
    if key not in direction_cache:
        direction_cache[key] = get_direction(src_prompt, tgt_prompt).to(device)
    direction = direction_cache[key]

    # === IDENTITY-PRESERVING EDIT ===
    edited = apply_identity_preserving_edit(latent, direction, layers=[4,5,6,7], strength=1.5, blend_ratio=0.3)

    try:
        with torch.no_grad():
            # Edited image
            img, _ = generator([edited], input_is_latent=True)
            img = (img.clamp(-1, 1) + 1) / 2.0
            img = img[0].cpu().permute(1, 2, 0).numpy() * 255
            img = img.astype('uint8')

            if save_comparison:
                orig_img, _ = generator([latent], input_is_latent=True)
                orig_img = (orig_img.clamp(-1, 1) + 1) / 2.0
                orig_img = orig_img[0].cpu().permute(1, 2, 0).numpy() * 255
                orig_img = orig_img.astype('uint8')

                comp = np.hstack((orig_img, img))
                Image.fromarray(comp).save(os.path.join(output_dir, "comp_" + fname))
            else:
                Image.fromarray(img).save(os.path.join(output_dir, fname))

            success_count += 1

    except Exception as e:
        print(f"❌ Failed rendering {fname}: {e}")
        skipped.append(fname)

print(f"\n✅ Completed: {success_count}/{len(df)}")
if skipped:
    print("⚠️ Skipped files:")
    for f in skipped:
        print("  -", f)
