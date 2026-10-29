"""Data loading for MedMNIST v2 datasets (PneumoniaMNIST, DermaMNIST)."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
import medmnist
from medmnist import INFO

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_split(flag, split, size, root):
    info = INFO[flag]
    cls = getattr(medmnist, info["python_class"])
    ds = cls(split=split, download=True, size=size, root=root)
    imgs = ds.imgs
    if imgs.ndim == 3:  # grayscale -> (N, H, W, 1)
        imgs = imgs[..., None]
    labels = ds.labels.squeeze(1).astype(np.int64)
    return imgs, labels


def load_dataset(flag, size, root):
    """Returns dict split -> (images uint8 NHWC, labels int64) and the class names."""
    splits = {s: load_split(flag, s, size, root) for s in ("train", "val", "test")}
    names = [v for _, v in sorted(INFO[flag]["label"].items(), key=lambda kv: int(kv[0]))]
    return splits, names


class MedImageDataset(Dataset):
    """Converts uint8 NHWC arrays to normalized 3-channel tensors (ImageNet stats)."""

    def __init__(self, imgs, labels, train, augment_flip):
        self.imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)  # NCHW uint8
        if self.imgs.shape[1] == 1:
            self.imgs = self.imgs.repeat(1, 3, 1, 1)
        self.labels = torch.from_numpy(labels)
        aug = []
        if train:
            aug.append(T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)))
            if augment_flip:
                aug += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
            aug.append(T.ColorJitter(brightness=0.1, contrast=0.1))
        self.transform = T.Compose(aug + [
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(self.imgs[i]), self.labels[i]


def make_loaders(splits, batch_size, augment_flip, num_workers=0):
    loaders = {}
    for s, (x, y) in splits.items():
        ds = MedImageDataset(x, y, train=(s == "train"), augment_flip=augment_flip)
        loaders[s] = DataLoader(ds, batch_size=batch_size, shuffle=(s == "train"),
                                num_workers=num_workers, pin_memory=num_workers > 0,
                                persistent_workers=num_workers > 0)
    return loaders


def class_weights(labels, n_classes):
    """Inverse-sqrt-frequency weights, normalized to mean 1 (softer than full inverse)."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = 1.0 / np.sqrt(counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)
