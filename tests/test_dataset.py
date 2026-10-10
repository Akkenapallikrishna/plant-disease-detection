import pytest
import torch
from PIL import Image

from dataset import AddGaussianNoise, ToColorMode, build_transforms, get_dataloaders


def test_eval_transform_output_shape():
    img = Image.new("RGB", (300, 250), color=(30, 120, 40))
    tensor = build_transforms(train=False)(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_color_mode_conversion():
    img = Image.new("RGB", (10, 10), color=(200, 50, 50))
    assert ToColorMode("rgb")(img).mode == "RGB"
    assert ToColorMode("hsv")(img).mode == "HSV"
    with pytest.raises(ValueError):
        ToColorMode("lab")


def test_noise_is_applied_to_training_only():
    def has_noise(t):
        return any(isinstance(step, AddGaussianNoise) for step in t.transforms)

    assert has_noise(build_transforms(train=True, noise=True))
    assert not has_noise(build_transforms(train=False, noise=True))
    assert not has_noise(build_transforms(train=True, noise=False))


def test_gaussian_noise_stays_in_valid_pixel_range():
    torch.manual_seed(0)
    noisy = AddGaussianNoise(std=0.5)(torch.rand(3, 32, 32))
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0


def test_dataloaders_on_fake_data(fake_data_dir):
    loaders, classes = get_dataloaders(fake_data_dir, batch_size=4)
    assert len(classes) == 3
    assert len(loaders["train"].dataset) == 21
    assert len(loaders["val"].dataset) == 6
    assert len(loaders["test"].dataset) == 3
    images, labels = next(iter(loaders["train"]))
    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)
