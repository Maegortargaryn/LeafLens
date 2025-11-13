# continue_train.py
"""
Load the latest best_model*.pth and continue fine-tuning with weight decay + ReduceLROnPlateau.
Run (example):
    python continue_train.py --epochs 8 --batch_size 16 --lr 1e-4 --num_workers 0
"""
import argparse, os, random
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_transforms(img_size=224):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tfms, val_tfms

def make_loaders(data_dir, batch_size=16, img_size=224, num_workers=0):
    train_tfms, val_tfms = get_transforms(img_size)
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    classes = ckpt.get("classes", None)
    return classes

def validate(model, loader, device, criterion):
    model.eval(); running_loss = 0.0; correct = 0; total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Device:", device)

    train_loader, val_loader, classes = make_loaders(args.data, args.batch_size, args.img_size, args.num_workers)
    print("Num classes:", len(classes))

    # build model and load best checkpoint
    model = models.mobilenet_v2(pretrained=False)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, len(classes))
    model = model.to(device)

    # find latest checkpoint
    ckpts = sorted(Path(args.ckpt_dir).glob("best_model*.pth"))
    assert ckpts, "No checkpoint found in models/ — run training first."
    latest = ckpts[-1]
    cls_from_ckpt = load_checkpoint(model, latest, device)
    if cls_from_ckpt:
        print("Loaded classes from checkpoint.")
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"FineTune {epoch}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = validate(model, val_loader, device, criterion)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)

        # scheduler step
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.ckpt_dir, f"continued_best_epoch{epoch}_acc{val_acc:.4f}.pth")
            torch.save({"model_state_dict": model.state_dict(), "classes": classes}, save_path)
            print("Saved continued best to", save_path)

    writer.close()
    print("Fine-tuning complete. Best val acc:", best_val_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset_split")
    parser.add_argument("--ckpt_dir", default="models")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_dir", default="runs_finetune")
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()
    main(args)
