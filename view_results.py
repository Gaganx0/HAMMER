import streamlit as st
import json
from PIL import Image
import os
from html import escape
import cv2
import numpy as np

st.set_page_config(layout="wide")
st.title("HAMMER Prediction Viewer")

# Load JSON
json_path = "C:/Users/deepg/Documents/REIT_4842/code/MultiModal-DeepFake/per_sample_results.json"
if not os.path.exists(json_path):
    st.error("JSON file not found.")
    st.stop()

with open(json_path, "r") as f:
    results = json.load(f)

# Display each result
for i, sample in enumerate(results):
    st.markdown("---")
    cols = st.columns([1, 3])

    # Load and show image if path known
    img_path = sample.get("image_path") or sample.get("image")  # adjust based on your output
    if img_path and os.path.exists(img_path):
        image = Image.open(img_path)
        def draw_bbox_on_image(image_path, box_norm):
            image = cv2.imread(image_path)
            if image is None:
                return None

            h, w = image.shape[:2]
            cx, cy, bw, bh = box_norm
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # Draw rectangle (red)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Convert to RGB PIL image for Streamlit
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_rgb)

        # Use box if not empty
        box = sample.get("manipulated_image_box", [0, 0, 0, 0])
        if sum(box) > 0:
            img = draw_bbox_on_image(sample["image"], box)
        else:
            img = Image.open(sample["image"]).convert("RGB")

        cols[0].image(img.resize((256, 256)))
    else:
        cols[0].write("🚫 No image")

    # Show prediction details
    with cols[1]:
        def highlight_text(text, tokens):
            words = text.split()
            highlighted = []

            for i, word in enumerate(words):
                if word in tokens:
                   highlighted.append(f"<span style='color: #FFD700; font-weight: bold'>{escape(word)}</span>")

                else:
                    highlighted.append(escape(word))
            return " ".join(highlighted)

        highlighted = highlight_text(sample["text"], sample.get("manipulated_text_tokens", []))
        st.markdown(f"<strong>Text:</strong> {highlighted}", unsafe_allow_html=True)

        st.write(f"**Prediction:** {'MANIPULATED' if sample['is_manipulated'] else 'REAL'}")
        st.write(f"**Manipulation Probability:** {sample['manipulation_prob']}")
        st.write(f"**Types:** {', '.join(sample['manipulation_types']) or 'None'}")
        # st.write(f"**Manipulated Text Tokens:** {', '.join(sample['manipulated_text_tokens']) or 'None'}")
        if "iou_score" in sample:
            st.write(f"📏 **IOU:** {sample['iou_score']}")
        if "type_confidences" in sample:
            confidences = ', '.join([f"{k}: {v:.2f}" for k, v in sample["type_confidences"].items()])
            st.markdown(f"**Type Confidences:** {confidences}")

        if "token_scores" in sample and sample["token_scores"]:
            token_scores = ', '.join([f"{k}: {v:.2f}" for k, v in sample["token_scores"].items()])
            st.markdown(f"**Token Scores:** {token_scores}")

