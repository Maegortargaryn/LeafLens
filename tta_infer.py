# tta_infer.py
import torch, numpy as np
from PIL import Image
from torchvision import transforms, models
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--model", default="models/best_model_continued.pth")
parser.add_argument("--tta", type=int, default=6)  # number of transforms
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(args.model, map_location=device)
classes = ckpt["classes"]

model = models.mobilenet_v2(pretrained=False)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, len(classes))
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

base_tf = transforms.Compose([
    transforms.Resize(int(224*1.15)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

img = Image.open(args.image).convert("RGB")

# small TTA set
transforms_list = [
    lambda x: x,
    lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
    lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
    lambda x: x.rotate(15, expand=True),
    lambda x: x.rotate(-15, expand=True),
]

probs = np.zeros((len(classes),), dtype=np.float32)
for t in transforms_list:
    im2 = t(img)
    x = base_tf(im2).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        p = torch.softmax(out, dim=1)[0].cpu().numpy()
    probs += p

probs /= len(transforms_list)
idx = probs.argmax()
print("Predicted:", classes[idx], f"({probs[idx]*100:.2f}%)")
print("Top3:", sorted([(classes[i], float(probs[i])) for i in range(len(classes))], key=lambda x: -x[1])[:3])
