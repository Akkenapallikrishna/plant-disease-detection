import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import datasets

from dataset import build_transforms
from evaluate import AddRGBNoise, compute_metrics, short_name
from gradcam import preprocess


def test_perfect_predictions_score_one():
    labels = np.array([0, 1, 2, 3, 4, 5])
    probs = np.eye(6)[labels] * 0.9 + 0.1 / 6
    metrics = compute_metrics(probs, labels)
    assert metrics == pytest.approx({"top1": 1.0, "top5": 1.0, "macro_f1": 1.0, "roc_auc": 1.0})


def test_top5_counts_correct_class_ranked_second():
    # True class is always ranked 2nd -> wrong at Top-1, right at Top-5.
    labels = np.array([0, 1, 2, 3, 4, 5])
    probs = np.full((6, 6), 0.05)
    for i, label in enumerate(labels):
        probs[i, label] = 0.3
        probs[i, (label + 1) % 6] = 0.5
    metrics = compute_metrics(probs, labels)
    assert metrics["top1"] == 0.0
    assert metrics["top5"] == 1.0


def test_short_name():
    assert short_name("Tomato__Tomato_YellowLeaf__Curl_Virus") == "Tomato YellowLeaf Curl Virus"
    assert short_name("Pepper__bell___Bacterial_spot") == "Pepper Bacterial spot"
    assert short_name("Potato___healthy") == "Potato healthy"


@pytest.mark.parametrize("color_mode", ["rgb", "hsv"])
def test_app_preprocessing_matches_evaluation(fake_data_dir, color_mode):
    """Regression test for training-serving skew: the app/Grad-CAM path must produce
    exactly the same tensor as the evaluation pipeline (an earlier PIL resize did not)."""
    ds = datasets.ImageFolder(fake_data_dir / "test", build_transforms(train=False, color_mode=color_mode))
    for (path, _), (expected, _) in zip(ds.samples, ds):
        tensor, display = preprocess(Image.open(path), color_mode)
        assert torch.equal(tensor[0], expected)
        assert display.shape == (224, 224, 3)


def test_noisy_test_set_is_identical_after_reset():
    img = Image.new("RGB", (32, 32), color=(100, 150, 50))
    noise = AddRGBNoise(std=0.05, seed=1)
    first = np.asarray(noise(img))
    noise.reset(1)
    second = np.asarray(noise(img))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, np.asarray(img))
