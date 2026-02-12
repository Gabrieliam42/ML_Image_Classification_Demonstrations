import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageFile
from sklearn.cluster import KMeans
from train_runtime_utils import (
    build_runtime_config,
    configure_torch_backend,
    log_and_check_peak_vram,
    make_autocast,
    maybe_compile_model,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = os.getenv("DATA_DIR", "imgdata")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
WORKERS = int(os.getenv("WORKERS", "4"))
USE_PRETRAINED = os.getenv("USE_PRETRAINED", "1").strip().lower() in {"1", "true", "yes", "on"}
NUM_CLUSTERS = int(os.getenv("NUM_CLUSTERS", "3"))
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

class FlatImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder)
                      if os.path.splitext(f)[1].lower() in IMG_EXTS]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def prepare_data(folder, batch_size):
    transform = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE*1.1)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    dataset = FlatImageDataset(folder, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=WORKERS > 0,
        prefetch_factor=4 if WORKERS > 0 else None,
    )
    return loader

def build_feature_extractor(device):
    weights = ResNet18_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None
    model = resnet18(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove fc layer
    model = model.to(device)
    model.eval()
    return model

def extract_features(loader, model, device, use_amp):
    features = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            with make_autocast(device, use_amp):
                out = model(xb)
            out = out.view(out.size(0), -1)
            features.append(out.detach())
    return torch.cat(features, dim=0)

def torch_kmeans_gpu(feats: torch.Tensor, n_clusters: int, seed: int, max_iter: int = 100, tol: float = 1e-4):
    g = torch.Generator(device=feats.device)
    g.manual_seed(seed)

    n_samples = feats.size(0)
    init_idx = torch.randperm(n_samples, generator=g, device=feats.device)[:n_clusters]
    centers = feats[init_idx].clone()
    labels = torch.zeros(n_samples, dtype=torch.long, device=feats.device)
    prev_shift = None

    for _ in range(max_iter):
        dists = torch.cdist(feats, centers)
        labels = torch.argmin(dists, dim=1)

        new_centers = []
        for k in range(n_clusters):
            mask = labels == k
            if torch.any(mask):
                new_centers.append(feats[mask].mean(dim=0))
            else:
                ridx = torch.randint(0, n_samples, (1,), generator=g, device=feats.device)
                new_centers.append(feats[ridx].squeeze(0))
        new_centers = torch.stack(new_centers, dim=0)

        shift = torch.norm(new_centers - centers, dim=1).mean()
        centers = new_centers
        if prev_shift is not None and torch.abs(prev_shift - shift) < tol:
            break
        prev_shift = shift

    return labels.cpu().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    runtime = build_runtime_config(device)
    configure_torch_backend(device)
    log(
        f"Runtime: amp={runtime['use_amp']} compile={runtime['use_compile']} "
        f"adaptive_batch={runtime['adaptive_batch']} max_vram_gb={runtime['max_vram_gb']}"
    )
    log(
        f"Config: batch={BATCH_SIZE} image_size={IMAGE_SIZE} workers={WORKERS} "
        f"clusters={NUM_CLUSTERS} pretrained={USE_PRETRAINED}"
    )

    effective_batch = BATCH_SIZE
    if device.type == "cuda" and runtime["adaptive_batch"]:
        props = torch.cuda.get_device_properties(device)
        max_bytes = int(runtime["max_vram_gb"] * (1024 ** 3))
        scale = min(1.0, max_bytes / max(props.total_memory, 1))
        effective_batch = max(1, int(BATCH_SIZE * scale))
        if effective_batch != BATCH_SIZE:
            log(f"Adaptive batch: requested={BATCH_SIZE}, using={effective_batch}")

    loader = prepare_data(DATA_DIR, effective_batch)
    model = build_feature_extractor(device)
    model = maybe_compile_model(model, runtime["use_compile"], log)
    log("Extracting features...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    feats = extract_features(loader, model, device, runtime["use_amp"])

    use_gpu_kmeans = False
    cluster_ids = None
    try:
        import cupy as cp
        from cuml.cluster import KMeans as CuKMeans

        use_gpu_kmeans = device.type == "cuda"
        if use_gpu_kmeans:
            if hasattr(cp, "from_dlpack"):
                feat_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(feats))
            else:
                feat_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(feats))
            log(f"Clustering {feat_cp.shape[0]} samples into {NUM_CLUSTERS} clusters (cuML/GPU)...")
            kmeans = CuKMeans(n_clusters=NUM_CLUSTERS, random_state=42)
            cluster_ids = cp.asnumpy(kmeans.fit_predict(feat_cp))
    except Exception:
        use_gpu_kmeans = False

    if not use_gpu_kmeans and device.type == "cuda":
        log(f"Clustering {feats.shape[0]} samples into {NUM_CLUSTERS} clusters (torch/GPU fallback)...")
        cluster_ids = torch_kmeans_gpu(feats, NUM_CLUSTERS, 42)
        use_gpu_kmeans = True

    if not use_gpu_kmeans:
        feats_np = feats.cpu().numpy()
        log(f"Clustering {feats_np.shape[0]} samples into {NUM_CLUSTERS} clusters (sklearn/CPU)...")
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
        cluster_ids = kmeans.fit_predict(feats_np)

    log_and_check_peak_vram(
        device=device,
        max_vram_gb=runtime["max_vram_gb"],
        strict=runtime["strict_vram"],
        log=log,
    )

    for i, cluster in enumerate(cluster_ids):
        print(f"Image {i} -> Cluster {cluster}")

if __name__=="__main__":
    main()
