# infer.py
import torch
from torchvision import transforms, models
from PIL import Image
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--model", default=None)  # if None uses latest best
args = parser.parse_args()

MODEL_PATH = args.model or sorted(Path("models").glob("best_model*.pth"))[-1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_tfms = transforms.Compose([
    transforms.Resize(int(224*1.15)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

ckpt = torch.load(str(MODEL_PATH), map_location=device)
classes = ckpt["classes"]

model = models.mobilenet_v2(pretrained=False)
model.classifier[1].out_features = len(classes)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, len(classes))
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

img = Image.open(args.image).convert("RGB")
x = val_tfms(img).unsqueeze(0).to(device)
with torch.no_grad():
    out = model(x)
    probs = torch.softmax(out, dim=1)[0].cpu().numpy()

idx = int(probs.argmax())
print(f"Predicted: {classes[idx]} ({probs[idx]*100:.1f}%)")
