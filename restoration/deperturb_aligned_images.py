import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from glob import glob

# === CONFIG ===
input_dir = "./images/aligned"  # Folder with 256x256 aligned faces
overwrite = True  # Overwrite files in-place

# === Perturbation Detection ===
def is_blurry(img, threshold=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def is_jpeg_compressed(img, ssim_threshold=0.95):
    _, encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    decoded = cv2.imdecode(encoded, 1)
    sim, _ = ssim(img, decoded, full=True, channel_axis=-1)
    return sim < ssim_threshold


def is_perturbed(img):
    return is_blurry(img) or is_jpeg_compressed(img)

# === Deperturbation Pipeline ===
def deperturb(img):
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened

# === Main Processing Loop ===
image_paths = glob(os.path.join(input_dir, "*.jpg"))

print(f"🔍 Checking {len(image_paths)} images in {input_dir}...")

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Failed to read: {path}")
        continue

    if is_perturbed(img):
        fixed = deperturb(img)
        cv2.imwrite(path, fixed)
        print(f"🧽 Deperturbed and saved: {os.path.basename(path)}")
    else:
        print(f"✅ Already clean: {os.path.basename(path)}")
