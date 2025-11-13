import os
import shutil
import random

def split_data(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    classes = [c for c in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, c))]

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        random.shuffle(images)

        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split_name, split_files in splits.items():
            split_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)

            for img_name in split_files:
                src = os.path.join(cls_path, img_name)
                dst = os.path.join(split_dir, img_name)
                shutil.copyfile(src, dst)

        print(f"✔️ Done splitting class: {cls}")

    print("\n🎉 Dataset split completed!")
    print(f"Saved inside: {output_dir}")


if __name__ == "__main__":
    split_data("data", "dataset_split")
