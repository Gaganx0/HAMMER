import os
import json
import cv2

input_json = "per_sample_results_FA_only.json"
output_dir = "./images/aligned"
restored_dir = "./restored_images"
os.makedirs(output_dir, exist_ok=True)

with open(input_json, "r") as f:
    data = json.load(f)

for entry in data:
    # Extract original image name (without _out)
    original_name = os.path.splitext(os.path.basename(entry["image"]))[0]
    restored_name = f"{original_name}_out.jpg"
    image_path = os.path.join(restored_dir, restored_name)

    box = entry["manipulated_image_box"]

    if not os.path.exists(image_path):
        print(f"⚠️ Image not found: {image_path}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read: {image_path}")
        continue

    print(f"📂 Processing: {image_path}")

    h, w = image.shape[:2]
    x_center, y_center, box_w, box_h = box
    x_center *= w
    y_center *= h
    box_w *= w * 1.5
    box_h *= h * 1.5

    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    face_crop = image[y1:y2, x1:x2]
    if face_crop.size == 0:
        print(f"⚠️ Empty crop skipped: {image_path}")
        continue

    face_crop = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    # Save with original filename (no _out)
    save_path = os.path.join(output_dir, original_name + ".jpg")
    cv2.imwrite(save_path, face_crop)
    print(f"   💾 Saved: {save_path}")
