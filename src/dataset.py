"""
Data pipeline: turns leaf images into normalized tensors the CNN can read.

Key design decisions (good interview talking points):
- 224x224 input: the size ResNet50 / DenseNet121 were pretrained on (ImageNet).
- color_mode switch ("rgb" or "hsv") so we can run a controlled RGB-vs-HSV comparison.
- Augmentation (flips, rotation, optional Gaussian noise) is applied to TRAIN ONLY.
  Val/test images are never augmented, so their scores reflect real performance.
- ImageNet mean/std normalization, because the pretrained weights expect inputs
  on that scale.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
COLOR_MODES = ("rgb", "hsv")


class ToColorMode:
    """Convert a PIL image to RGB or HSV.

    HSV separates colour (Hue) from brightness (Value), which can make
    yellowing/browning of leaves easier to isolate than in RGB.
    """

    def __init__(self, mode: str = "rgb"):
        if mode not in COLOR_MODES:
            raise ValueError(f"color_mode must be one of {COLOR_MODES}, got {mode!r}")
        self.mode = mode

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        return img.convert("HSV") if self.mode == "hsv" else img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mode={self.mode!r})"


class AddGaussianNoise:
    """Add random Gaussian noise to a [0, 1] tensor to mimic smartphone camera grain."""

    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"


def build_transforms(train: bool, color_mode: str = "rgb", noise: bool = False) -> transforms.Compose:
    """Return the preprocessing pipeline for one split.

    Order matters: colour conversion -> resize/augment (PIL) -> ToTensor -> noise -> normalize.
    Noise is added before normalization so `std` is on the intuitive 0-1 pixel scale.
    """
    steps = [ToColorMode(color_mode), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]

    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
        ]

    steps.append(transforms.ToTensor())  # PIL [0, 255] -> float tensor [0, 1], shape (3, H, W)

    if train and noise:
        steps.append(AddGaussianNoise(std=0.05))

    steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(steps)


def get_dataloaders(
    data_dir: Path = DEFAULT_DATA_DIR,
    color_mode: str = "rgb",
    noise: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,  # 0 is the safe choice on Windows
    seed: int = 42,
):
    """Build train/val/test DataLoaders. Returns (loaders_dict, class_names)."""
    data_dir = Path(data_dir)
    splits = {
        "train": datasets.ImageFolder(data_dir / "train", build_transforms(True, color_mode, noise)),
        "val": datasets.ImageFolder(data_dir / "val", build_transforms(False, color_mode)),
        "test": datasets.ImageFolder(data_dir / "test", build_transforms(False, color_mode)),
    }

    # ImageFolder sorts class folders alphabetically; all three splits must agree.
    class_names = splits["train"].classes
    for name, ds in splits.items():
        if ds.classes != class_names:
            raise ValueError(f"Class folders in '{name}' don't match 'train'.")

    generator = torch.Generator().manual_seed(seed)  # reproducible shuffling
    loaders = {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),  # only shuffle training data
            num_workers=num_workers,
            generator=generator if name == "train" else None,
        )
        for name, ds in splits.items()
    }
    return loaders, class_names


if __name__ == "__main__":
    # Quick sanity check: python src/dataset.py
    for mode in COLOR_MODES:
        loaders, classes = get_dataloaders(color_mode=mode, noise=True)
        images, labels = next(iter(loaders["train"]))
        print(f"[{mode.upper()}] classes={len(classes)}  "
              f"train={len(loaders['train'].dataset)}  val={len(loaders['val'].dataset)}  "
              f"test={len(loaders['test'].dataset)}")
        print(f"       batch shape={tuple(images.shape)}  labels shape={tuple(labels.shape)}  "
              f"pixel range=[{images.min():.2f}, {images.max():.2f}]")
    print("\nClass index mapping:")
    for i, name in enumerate(classes):
        print(f"  {i:>2}: {name}")
