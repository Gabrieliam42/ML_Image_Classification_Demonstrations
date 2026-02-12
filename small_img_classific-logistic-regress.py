# Script Developer: Gabriel Mihai Sandu
# GitHub Profile: https://github.com/Gabrieliam42

import csv
import os, time, math
from collections import Counter, defaultdict
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms, datasets
from PIL import Image, ImageFile, features
from train_runtime_utils import (
    adaptive_batch_size_image,
    build_runtime_config,
    configure_torch_backend,
    log_and_check_peak_vram,
    make_autocast,
    make_grad_scaler,
    maybe_compile_model,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = os.getenv("DATA_DIR", "imgdata")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "imgcheckpoints")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "5"))
LR = float(os.getenv("LR", "1e-4"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.2"))
WORKERS = int(os.getenv("WORKERS", "4"))
PRINT_EVERY_BATCH = 1
SEED = 42
MIN_CLASS_SAMPLES = int(os.getenv("MIN_CLASS_SAMPLES", "3"))
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
OPENIMAGES_ANN_FILE = "train-annotations-bbox.csv"
OPENIMAGES_CLASS_FILE = "class-descriptions-boxable.csv"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def find_image_files(root: Path):
    files = []
    log(f"Searching for image files in {root}")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
            log(f"  Found image: {p}")
    log(f"Total images found: {len(files)}")
    return sorted(files)

class FlatImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.targets = [label for _, label in samples]
        log(f"FlatImageDataset created with {len(samples)} labeled files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y

def _build_transform():
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.1)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def _find_metadata_file(data_path: Path, filename: str):
    candidates = [
        data_path / filename,
        data_path.parent / filename,
        Path.cwd() / filename,
    ]
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if p.exists() and p.is_file():
            return p
    return None

def _load_openimages_label_map(image_ids, data_path: Path):
    ann_path = _find_metadata_file(data_path, OPENIMAGES_ANN_FILE)
    class_path = _find_metadata_file(data_path, OPENIMAGES_CLASS_FILE)
    if ann_path is None or class_path is None:
        log("OpenImages metadata files not found. "
            f"Expected {OPENIMAGES_ANN_FILE} and {OPENIMAGES_CLASS_FILE}.")
        return {}

    log(f"Loading OpenImages class map from {class_path}")
    class_id_to_name = {}
    with class_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                class_id_to_name[row[0]] = row[1]

    image_ids = set(image_ids)
    remaining = set(image_ids)
    image_id_to_label = {}
    log(f"Scanning OpenImages annotations at {ann_path} for {len(image_ids)} image IDs")
    with ann_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"ImageID", "LabelName"}
        if not reader.fieldnames or not needed.issubset(set(reader.fieldnames)):
            log(f"Annotations CSV missing required columns. Found: {reader.fieldnames}")
            return {}
        for row in reader:
            image_id = row.get("ImageID")
            if image_id in remaining:
                label_id = row.get("LabelName", "")
                label_name = class_id_to_name.get(label_id, label_id)
                image_id_to_label[image_id] = label_name
                remaining.remove(image_id)
                if not remaining:
                    break

    log(f"OpenImages labels matched: {len(image_id_to_label)}/{len(image_ids)}")
    return image_id_to_label

def _prepare_labeled_flat_samples(files, data_path: Path):
    image_ids = [p.stem for p in files]
    id_to_label = _load_openimages_label_map(image_ids, data_path)
    if not id_to_label:
        raise ValueError(
            "Flat image folder has no class labels. "
            "Use class subfolders (ImageFolder format) or provide OpenImages metadata CSV files."
        )

    labeled = []
    unmatched = 0
    for p in files:
        label = id_to_label.get(p.stem)
        if label is None:
            unmatched += 1
            continue
        labeled.append((p, label))

    if unmatched:
        log(f"Dropping {unmatched} images without metadata labels")

    if not labeled:
        raise ValueError("No labeled images found after metadata matching.")

    label_counts = Counter(label for _, label in labeled)
    keep_labels = {label for label, c in label_counts.items() if c >= MIN_CLASS_SAMPLES}
    dropped_rare = len(labeled) - sum(label_counts[k] for k in keep_labels)
    if dropped_rare:
        log(f"Dropping {dropped_rare} images from classes with < {MIN_CLASS_SAMPLES} samples")

    filtered = [(p, label) for p, label in labeled if label in keep_labels]
    if not filtered:
        raise ValueError(
            f"All labeled images were filtered out by MIN_CLASS_SAMPLES={MIN_CLASS_SAMPLES}."
        )

    class_names = sorted({label for _, label in filtered})
    if len(class_names) < 2:
        raise ValueError(
            f"Need at least 2 classes for classification. Found {len(class_names)} class."
        )

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    indexed_samples = [(p, class_to_idx[label]) for p, label in filtered]

    log(f"Flat folder labeled via OpenImages metadata: {len(indexed_samples)} images, "
        f"{len(class_names)} classes")
    top_counts = Counter([y for _, y in indexed_samples]).most_common(10)
    top_desc = ", ".join([f"{class_names[idx]}:{cnt}" for idx, cnt in top_counts])
    log(f"Top classes: {top_desc}")
    return indexed_samples, class_names

def _stratified_split(dataset, val_split: float):
    targets = list(getattr(dataset, "targets", []))
    if not targets:
        return None, None

    class_to_indices = defaultdict(list)
    for idx, y in enumerate(targets):
        class_to_indices[int(y)].append(idx)

    g = torch.Generator().manual_seed(SEED)
    train_indices, val_indices = [], []
    for _, idxs in class_to_indices.items():
        order = torch.randperm(len(idxs), generator=g).tolist()
        idxs = [idxs[i] for i in order]
        per_class_val = int(math.floor(val_split * len(idxs)))
        if len(idxs) >= 2:
            per_class_val = max(1, min(len(idxs) - 1, per_class_val))
        else:
            per_class_val = 0
        val_indices.extend(idxs[:per_class_val])
        train_indices.extend(idxs[per_class_val:])

    if not val_indices:
        return None, None

    train_order = torch.randperm(len(train_indices), generator=g).tolist()
    val_order = torch.randperm(len(val_indices), generator=g).tolist()
    train_indices = [train_indices[i] for i in train_order]
    val_indices = [val_indices[i] for i in val_order]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def prepare_datasets(data_dir):
    data_path = Path(data_dir)
    log(f"Preparing datasets from {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    webp_ok = features.check("webp")
    if not webp_ok:
        log("WARNING: Pillow compiled without WEBP support. .webp files may fail to open.")

    subdirs = [p for p in data_path.iterdir() if p.is_dir()]
    has_subfolders_with_images = False
    for d in subdirs:
        imgs = find_image_files(d)
        if imgs:
            has_subfolders_with_images = True

    if has_subfolders_with_images:
        log("Detected class subfolders. Using ImageFolder dataset.")
        transform = _build_transform()
        dataset = datasets.ImageFolder(root=str(data_path), transform=transform)
        class_names = dataset.classes
        total_files = len(dataset)
        log(f"Classes detected: {class_names}, total images: {total_files}")
    else:
        files = [p for p in data_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not files:
            files = find_image_files(data_path)
        if not files:
            raise FileNotFoundError(f"No image files found under {data_dir}.")
        log(f"Flat folder detected with {len(files)} images")
        transform = _build_transform()
        samples, class_names = _prepare_labeled_flat_samples(files, data_path)
        dataset = FlatImageDataset(samples, transform=transform)
        total_files = len(dataset)

    if len(class_names) < 2:
        raise ValueError(f"Need at least 2 classes to train a classifier. Found {len(class_names)}.")

    val_size = int(math.floor(VAL_SPLIT * total_files))
    train_size = total_files - val_size
    if val_size == 0:
        log("Validation split is 0, using train-only run")
        return dataset, None, class_names

    split_train, split_val = _stratified_split(dataset, VAL_SPLIT)
    if split_train is not None and split_val is not None:
        train_dataset, val_dataset = split_train, split_val
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    log(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    return train_dataset, val_dataset, class_names

def build_model(num_classes, device):
    log(f"Building logistic regression model with {num_classes} output classes")
    in_features = 3 * IMAGE_SIZE * IMAGE_SIZE
    model = nn.Linear(in_features, num_classes).to(device)
    log(f"Model on device: {device}")
    return model

def save_checkpoint(state, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save(state, path)
    log(f"Checkpoint saved: {path}")

def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    log(f"Current working directory: {os.getcwd()}")
    log(f"Data folder: {DATA_DIR}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device info: {device}, torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    runtime = build_runtime_config(device)
    configure_torch_backend(device)
    log(
        f"Runtime: amp={runtime['use_amp']} compile={runtime['use_compile']} "
        f"adaptive_batch={runtime['adaptive_batch']} max_vram_gb={runtime['max_vram_gb']}"
    )

    train_ds, val_ds, class_names = prepare_datasets(DATA_DIR)
    num_classes = len(class_names)
    log(f"Classes ({num_classes}): {class_names[:20]}{' ...' if num_classes > 20 else ''}")

    model = build_model(num_classes, device)
    model = maybe_compile_model(model, runtime["use_compile"], log)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: total={total_params}, trainable={trainable}")

    effective_batch = BATCH_SIZE
    if runtime["adaptive_batch"]:
        effective_batch = adaptive_batch_size_image(
            model=model,
            device=device,
            image_size=IMAGE_SIZE,
            num_classes=num_classes,
            requested_batch=BATCH_SIZE,
            use_amp=runtime["use_amp"],
            max_vram_gb=runtime["max_vram_gb"],
            reserve_mb=runtime["reserve_mb"],
            flatten_input=True,
            log=log,
        )

    log(f"Building dataloaders with batch_size={effective_batch}...")
    dl_kwargs = {
        "num_workers": WORKERS,
        "pin_memory": device.type == "cuda",
    }
    if WORKERS > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=effective_batch, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=effective_batch, shuffle=False, **dl_kwargs) if val_ds else None
    log(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader) if val_loader else 0}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = make_grad_scaler(runtime["use_amp"])

    log("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        running_loss = running_correct = running_total = 0

        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb = xb.view(xb.size(0), -1)
            optimizer.zero_grad(set_to_none=True)
            with make_autocast(device, runtime["use_amp"]):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            if runtime["use_amp"]:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * xb.size(0)
            running_correct += (preds == yb).sum().item()
            running_total += xb.size(0)

            log(f"Epoch {epoch} batch {batch_idx}/{len(train_loader)}: "
                f"batch_loss={loss.item():.4f}, batch_acc={(preds==yb).float().mean().item():.4f}, "
                f"running_loss={running_loss/running_total:.4f}, running_acc={running_correct/running_total:.4f}")

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        epoch_time = time.time() - epoch_start
        log(f"Epoch {epoch} complete: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, time={epoch_time:.1f}s")

        if val_loader:
            model.eval()
            val_loss = val_correct = val_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    xb = xb.view(xb.size(0), -1)
                    with make_autocast(device, runtime["use_amp"]):
                        logits = model(xb)
                        loss = loss_fn(logits, yb)
                    preds = logits.argmax(dim=1)
                    val_loss += loss.item() * xb.size(0)
                    val_correct += (preds==yb).sum().item()
                    val_total += xb.size(0)
                    log(f"  Val batch: loss={loss.item():.4f}, acc={(preds==yb).float().mean().item():.4f}")

            val_loss /= val_total
            val_acc = val_correct / val_total
            log(f"Validation: loss={val_loss:.4f}, acc={val_acc:.4f}, samples={val_total}")

        log_and_check_peak_vram(
            device=device,
            max_vram_gb=runtime["max_vram_gb"],
            strict=runtime["strict_vram"],
            log=log,
        )

        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "num_classes": num_classes,
            "class_names": class_names,
            "loss": epoch_loss,
            "acc": epoch_acc,
        }, epoch, CHECKPOINT_DIR)

    log("Training finished. Final evaluation:")
    model.eval()
    with torch.no_grad():
        sample_x, sample_y = next(iter(train_loader))
        sample_x = sample_x.to(device)
        sample_x = sample_x.view(sample_x.size(0), -1)
        sample_logits = torch.softmax(model(sample_x), dim=1)
        for i in range(min(6, sample_x.size(0))):
            top1 = sample_logits[i].argmax().item()
            conf = sample_logits[i, top1].item()
            log(f"  Sample {i}: true={sample_y[i]}, pred_class={top1}, conf={conf:.3f}")

if __name__ == "__main__":
    main()
    
# Script Developer: Gabriel Mihai Sandu
# GitHub Profile: https://github.com/Gabrieliam42
