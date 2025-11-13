# webapp/app.py
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import torch
from torchvision import transforms, models
from PIL import Image
import io
import os
import numpy as np

import json
from pathlib import Path

# Load plant info JSON
PLANT_INFO_PATH = Path("plants_info.json")
if PLANT_INFO_PATH.exists():
    with open(PLANT_INFO_PATH, "r", encoding="utf8") as f:
        plants_info = json.load(f)
else:
    plants_info = {}

# Config
MODEL_PATH = Path("../models/best_model_continued.pth")  # adjust if needed
UPLOAD_FOLDER = Path("uploads")
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
IMG_SIZE = 224

UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# load model once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading model on device:", device)
print("Model path:", MODEL_PATH.resolve())

ckpt = torch.load(str(MODEL_PATH), map_location=device)
classes = ckpt.get("classes", None)
if classes is None:
    raise RuntimeError("Saved checkpoint does not contain 'classes' key.")

# build same architecture used for training
model = models.mobilenet_v2(pretrained=False)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, len(classes))
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

# transform (same as validation)
val_tfms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT

def predict_image_tta(image_bytes, k=3):
    """
    Perform simple Test-Time Augmentation (TTA) and return top-k (label, prob) pairs.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # small set of TTA transforms (change or remove to trade speed/accuracy)
    tta_transforms = [
        lambda x: x,
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
        lambda x: x.rotate(15, expand=True),
        lambda x: x.rotate(-15, expand=True),
    ]

    probs_sum = np.zeros((len(classes),), dtype=np.float32)

    for t in tta_transforms:
        try:
            im2 = t(img)
        except Exception:
            im2 = img  # fallback to original if transform fails
        x = val_tfms(im2).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            p = torch.softmax(out, dim=1)[0].cpu().numpy()
        probs_sum += p

    probs_avg = probs_sum / len(tta_transforms)
    topk_idx = probs_avg.argsort()[-k:][::-1]
    results = [(classes[i], float(probs_avg[i])) for i in topk_idx]
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        f = request.files["file"]
        if f.filename == "":
            return redirect(request.url)
        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            save_path = UPLOAD_FOLDER / fname
            f.save(save_path)

            with open(save_path, "rb") as fh:
                image_bytes = fh.read()
            results = predict_image_tta(image_bytes, k=3)
            # Get top predicted plant (top-1)
            top_label = results[0][0] if results else None
            plant_detail = plants_info.get(top_label, {})

            # build file URL for template
            file_url = url_for('uploaded_file', filename=fname)
            return render_template(
                "index.html",
                filename=file_url,
                results=results,
                classes=classes,
                top_label=top_label,
                plant_detail=plant_detail
            )
        else:
            return render_template("index.html", error="Invalid file type. Use jpg/png.")
    return render_template("index.html")

# serve uploaded images from uploads/ directory
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    # run dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
