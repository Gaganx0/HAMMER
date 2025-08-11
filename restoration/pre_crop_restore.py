import os
import cv2
import numpy as np
import subprocess

os.makedirs("restored_images", exist_ok=True)

def estimate_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def estimate_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    M = gray.mean()
    return np.mean((gray - M) ** 2)

def pad_to_multiple_of_8(image):
    h, w = image.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return padded, (h, w)

def crop_to_original(image, original_shape):
    return image[:original_shape[0], :original_shape[1]]

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return

    blur_score = estimate_blur(img)
    noise_score = estimate_noise(img)
    print(f"📊 {os.path.basename(image_path)} — Blur: {blur_score:.1f}, Noise: {noise_score:.1f}")

    padded_img, original_shape = pad_to_multiple_of_8(img)
    padded_path = image_path.replace("original", "padded_temp")
    os.makedirs(os.path.dirname(padded_path), exist_ok=True)
    cv2.imwrite(padded_path, padded_img)

    restored_name = os.path.basename(image_path)
    restored_path = os.path.join("restored_images", restored_name)

    if blur_score < 100 and noise_score > 2000:
        print("🧠 Applying: Restormer (motion_deblurring.pth)")
        subprocess.run([
            "conda", "run", "-n", "restorm", "python", "Restormer/inference_restormer.py",
            "--task", "motion_deblurring",
            "--input_dir", os.path.dirname(padded_path),
            "--result_dir", "restored_images/",
            "--weights", "pretrained_models/motion_deblurring.pth"
        ])
    else:
        print("🧠 Applying: Real-ESRGAN")
        subprocess.run([
            "conda", "run", "-n", "esrgan", "python", "Real-ESRGAN/inference_realesrgan.py",
            "-i", padded_path,
            "-o", "restored_images/",
            "-n", "RealESRGAN_x4plus",
            "--model_path", "pretrained_models/RealESRGAN_x4plus.pth",
            "--outscale", "1", 
            "--fp32",
            "--tile", "128",
            "--tile_pad", "10"
        ])

    if os.path.exists(restored_path):
        restored = cv2.imread(restored_path)
        if restored is not None:
            cropped = crop_to_original(restored, original_shape)
            cv2.imwrite(restored_path, cropped)

# Process all images in images/original/
input_dir = "images/original/"
for fname in os.listdir(input_dir):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        process_image(os.path.join(input_dir, fname))
