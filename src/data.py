"""PlantVillage data pipeline: scan, stratified split, manifest, augment, loaders."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "color"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "split_manifest.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

EXPECTED_CLASSES = sorted([
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scan_samples(color_dir: Path | str) -> list[tuple[Path, str]]:
    color_dir = Path(color_dir).resolve()
    class_names = sorted(p.name for p in color_dir.iterdir() if p.is_dir())
    missing = set(EXPECTED_CLASSES) - set(class_names)
    extra = set(class_names) - set(EXPECTED_CLASSES)
    if missing or extra:
        raise ValueError(f"class mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    samples = []
    for cls in class_names:
        for p in sorted((color_dir / cls).iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                samples.append((p.resolve(), cls))
    return samples


def create_split_manifest(color_dir: Path | str = DEFAULT_DATA_DIR,
                          manifest_path: Path | str = DEFAULT_MANIFEST,
                          seed: int = 42) -> dict:
    color_dir = Path(color_dir).resolve()
    samples = scan_samples(color_dir)
    paths = [str(p.relative_to(color_dir)) for p, _ in samples]
    labels = [label for _, label in samples]

    tr_idx, rest_idx = train_test_split(
        np.arange(len(paths)), train_size=0.8, shuffle=True,
        random_state=seed, stratify=labels)
    rest_labels = [labels[i] for i in rest_idx]
    val_idx, test_idx = train_test_split(
        np.arange(len(rest_idx)), train_size=0.5, shuffle=True,
        random_state=seed, stratify=rest_labels)
    tr_paths = [paths[i] for i in tr_idx]
    val_paths = [paths[rest_idx[i]] for i in val_idx]
    test_paths = [paths[rest_idx[i]] for i in test_idx]

    assert not (set(tr_paths) & set(val_paths))
    assert not (set(tr_paths) & set(test_paths))
    assert not (set(val_paths) & set(test_paths))

    manifest = {
        "color_dir": str(color_dir),
        "seed": seed,
        "num_images": len(paths),
        "classes": sorted(set(labels)),
        "split": {"train": tr_paths, "val": val_paths, "test": test_paths},
        "per_class": {c: labels.count(c) for c in sorted(set(labels))},
    }
    Path(manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_manifest(manifest_path: Path | str = DEFAULT_MANIFEST) -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def resolve_color_dir(manifest: dict, manifest_path: Path | str = DEFAULT_MANIFEST) -> Path:
    recorded = Path(manifest.get("color_dir", ""))
    if recorded.exists():
        return recorded
    return Path(manifest_path).parent / "color"


def get_manifest(color_dir: Path | str = DEFAULT_DATA_DIR,
                 manifest_path: Path | str = DEFAULT_MANIFEST,
                 seed: int = 42) -> dict:
    mp = Path(manifest_path)
    if mp.exists():
        return load_manifest(mp)
    return create_split_manifest(color_dir, mp, seed)


def build_transforms(train: bool, size: int = 224) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class PlantVillageDataset(Dataset):
    def __init__(self, color_dir: Path | str, paths: list[str], transform=None):
        self.color_dir = Path(color_dir)
        self.paths = paths
        self.transform = transform
        classes = sorted(set(p.split("/")[0] for p in paths)) if paths else []
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel = self.paths[idx]
        cls = rel.split("/")[0]
        img = Image.open(self.color_dir / rel).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[cls]


def build_loaders(batch_size: int = 64, num_workers: int = 8, seed: int = 42,
                  manifest_path: Path | str = DEFAULT_MANIFEST,
                  color_dir: Path | str | None = None):
    manifest = load_manifest(manifest_path)
    if color_dir is None:
        color_dir = resolve_color_dir(manifest, manifest_path)
    else:
        color_dir = Path(color_dir)
    split = manifest["split"]
    train_ds = PlantVillageDataset(color_dir, split["train"], build_transforms(True))
    val_ds = PlantVillageDataset(color_dir, split["val"], build_transforms(False))
    test_ds = PlantVillageDataset(color_dir, split["test"], build_transforms(False))
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0), generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, manifest
