import os
import time
import math
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image, ImageFile, features

CWD = Path.cwd()
DATA_DIR = CWD / "imgdata"
CHECKPOINT_DIR = CWD / "imgcheckpoints"
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_EPOCHS = 5
LR = 1e-3
VAL_SPLIT = 0.2
WORKERS = 8
N_DOWNLOAD = 1000
IMAGES_MANIFEST_URL = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable.csv"
LABELS_MANIFEST_URL = "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv"
ANNOTATIONS_MANIFEST_URL = "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
ImageFile.LOAD_TRUNCATED_IMAGES = True

def log(msg: str):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def download_csv(url, local_path):
    if Path(local_path).exists():
        return str(local_path)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)
    return str(local_path)

def get_existing_images(data_dir):
    log("Counting images")
    data_path = Path(data_dir)
    safe_mkdir(data_path)
    files = [p for p in data_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return files

def download_image(session, url, out_path, max_retries=3):
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and resp.content:
                tmp = out_path.with_suffix(out_path.suffix + ".part")
                with open(tmp, "wb") as f:
                    f.write(resp.content)
                try:
                    Image.open(tmp).verify()
                except Exception:
                    tmp.unlink(missing_ok=True)
                    continue
                tmp.rename(out_path)
                return True
        except:
            time.sleep(0.5)
    return False

def ensure_openimages_subset(data_dir, n_images=N_DOWNLOAD, workers=8):
    data_path = Path(data_dir)
    safe_mkdir(data_path)
    manifest_path = CWD / "train-images-boxable.csv"
    download_csv(IMAGES_MANIFEST_URL, manifest_path)
    df = pd.read_csv(manifest_path)
    existing = get_existing_images(data_dir)
    existing_ids = set([p.stem for p in existing])
    remaining = n_images - len(existing)
    log(f"Existing images: {len(existing)}, need to download: {remaining}")
    if remaining <= 0:
        df = df[df['ImageID'].isin(existing_ids)]
        return df
    candidates = df[~df['ImageID'].isin(existing_ids)]
    selected = candidates.sample(n=remaining, random_state=42) if remaining < len(candidates) else candidates
    session = requests.Session()
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, row in selected.iterrows():
            img_id = row['ImageID']
            url = row['OriginalURL']
            ext = Path(url.split("?")[0]).suffix.lower() if Path(url.split("?")[0]).suffix.lower() in IMG_EXTS else ".jpg"
            out_path = data_path / f"{img_id}{ext}"
            futures.append(ex.submit(download_image, session, url, out_path))
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            pass
    all_files = get_existing_images(data_dir)
    all_ids = [p.stem for p in all_files]
    return df[df['ImageID'].isin(all_ids)]

def get_labels_for_images(image_df):
    classes_csv = CWD / "class-descriptions-boxable.csv"
    download_csv(LABELS_MANIFEST_URL, classes_csv)
    class_df = pd.read_csv(classes_csv, header=None, names=['LabelID', 'LabelName'])
    label_map = dict(zip(class_df.LabelID, class_df.LabelName))
    annotations_csv = CWD / "train-annotations-bbox.csv"
    download_csv(ANNOTATIONS_MANIFEST_URL, annotations_csv)
    ann_df = pd.read_csv(annotations_csv)
    merged = ann_df[ann_df['ImageID'].isin(image_df['ImageID'])]
    first_labels = merged.groupby('ImageID').first().reset_index()
    first_labels['LabelName'] = first_labels['LabelName'].map(label_map)
    label_dict = dict(zip(first_labels['ImageID'], first_labels['LabelName']))
    return label_dict

class OpenImagesDataset(Dataset):
    def __init__(self, data_dir, image_label_dict, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_label_dict = image_label_dict
        self.files = [p for p in self.data_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        self.class_names = sorted(list(set(image_label_dict.values())))
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label_name = self.image_label_dict.get(p.stem, None)
        label_idx = self.class_to_idx[label_name] if label_name in self.class_to_idx else 0
        return img, label_idx

def prepare_dataloader(dataset):
    val_size = int(math.floor(VAL_SPLIT * len(dataset)))
    train_size = len(dataset) - val_size
    if val_size == 0:
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS), None
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    return train_loader, val_loader

def build_model(num_classes, device):
    from torchvision.models import ResNet18_Weights
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def main():
    log(f"CWD: {CWD}")
    safe_mkdir(DATA_DIR)
    safe_mkdir(CHECKPOINT_DIR)
    image_df = ensure_openimages_subset(DATA_DIR, N_DOWNLOAD)
    label_dict = get_labels_for_images(image_df)
    transform = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE*1.1)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    dataset = OpenImagesDataset(DATA_DIR, label_dict, transform=transform)
    train_loader, val_loader = prepare_dataloader(dataset)
    log(f"Classes: {dataset.class_names}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(dataset.class_names), device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(1)
            running_loss += loss.item()*xb.size(0)
            running_correct += (preds==yb).sum().item()
            total += xb.size(0)
        log(f"Epoch {epoch}  loss: {running_loss/total:.4f}  acc: {running_correct/total:.4f}")
        checkpoint_path = CHECKPOINT_DIR / f"epoch{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        log(f"Saved checkpoint {checkpoint_path}")

if __name__=="__main__":
    main()
