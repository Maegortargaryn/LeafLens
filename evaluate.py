# evaluate.py
import torch, os
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = "dataset_split"
MODEL_PATH = sorted(Path("models").glob("best_model*.pth"))[-1]  # latest best model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Loading model:", MODEL_PATH)

# transforms must match validation transforms used in training
val_tfms = transforms.Compose([
    transforms.Resize(int(224*1.15)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_tfms)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# build model same as train: mobilenet_v2
model = models.mobilenet_v2(pretrained=False)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, len(test_ds.classes))
ckpt = torch.load(str(MODEL_PATH), map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

acc = (all_preds == all_targets).mean()
print(f"Test accuracy: {acc:.4f}")

# confusion matrix and classification report
cm = confusion_matrix(all_targets, all_preds)
print("\nClassification report:\n")
print(classification_report(all_targets, all_preds, target_names=test_ds.classes))

# plot confusion matrix
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_ds.classes, yticklabels=test_ds.classes, cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"Confusion matrix — Test acc {acc:.4f}")
plt.tight_layout()
plt.show()
