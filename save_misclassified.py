# save_misclassified.py
import os
from pathlib import Path
import shutil
import torch
import numpy as np
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

DATA_DIR = "dataset_split"
OUT_DIR = "misclassified"
MODEL_PATH = sorted(Path("models").glob("best_model*.pth"))[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Using model:", MODEL_PATH)

val_tfms = transforms.Compose([
    transforms.Resize(int(224*1.15)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_tfms)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# build model same as training
model = models.mobilenet_v2(pretrained=False)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, len(test_ds.classes))
ckpt = torch.load(str(MODEL_PATH), map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
classes = ckpt.get("classes", test_ds.classes)
model.to(device).eval()

# ensure output folder
out_path = Path(OUT_DIR)
if out_path.exists():
    # clear previous results
    for f in out_path.iterdir():
        if f.is_file():
            f.unlink()
else:
    out_path.mkdir(parents=True)

# iterate and collect misclassified filenames
mis_count = 0
with torch.no_grad():
    for i, (imgs, labels) in enumerate(test_loader):
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        labels = labels.numpy()

        # get original file paths from dataset samples
        # ImageFolder stores (path, class_idx) pairs in test_ds.samples
        start = i * test_loader.batch_size
        for j in range(len(labels)):
            idx = start + j
            if idx >= len(test_ds.samples):
                continue
            img_path, true_idx = test_ds.samples[idx]
            pred_idx = int(preds[j])
            true_idx = int(labels[j])
            if pred_idx != true_idx:
                mis_count += 1
                true_name = classes[true_idx].replace(" ", "_")
                pred_name = classes[pred_idx].replace(" ", "_")
                dst_name = f"{Path(img_path).stem}__true-{true_name}__pred-{pred_name}{Path(img_path).suffix}"
                shutil.copy(img_path, out_path / dst_name)

print(f"Done. Misclassified images saved to: {out_path}  (count = {mis_count})")
