"""
Evaluate every trained model ONCE on the held-out test set.

Principles (good interview talking points):
- The final model is chosen by VALIDATION accuracy, before looking at test results,
  so the test score stays an honest estimate of real performance.
- Metrics: Top-1 accuracy, Top-5 accuracy, macro-F1 (treats every class equally)
  and macro one-vs-rest ROC-AUC (how well the model ranks the right class).
- Robustness test: every model is also evaluated on test images with Gaussian noise
  added in RGB pixel space (like real camera grain). This is the fair way to check
  whether noise-augmented training actually buys robustness.

Outputs (in results/):
    summary.csv, summary.md                 one row per model
    confusion_matrix_<run>.png              for the selected model
    classification_report_<run>.txt         per-class precision/recall/F1
    curves_<run>.png                        training curves for every run

Usage (from the project folder):
    python src/evaluate.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (  # noqa: E402
    DEFAULT_DATA_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ToColorMode,
)
from models import build_model  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_NOISE_STD = 0.05
NOISE_SEED = 1234


class AddRGBNoise:
    """Add Gaussian noise to the RGB image BEFORE any colour conversion.

    Real camera grain happens in RGB pixel space, so for HSV models we add the noise
    first and convert afterwards. A fixed seed makes the noisy test set identical for
    every model, so the comparison is fair.
    """

    def __init__(self, std: float, seed: int):
        self.std = std
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        arr = np.clip(arr + self.rng.normal(0.0, self.std, arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))


def eval_transform(color_mode: str, noise: AddRGBNoise | None = None) -> transforms.Compose:
    steps = [noise] if noise is not None else []
    steps += [
        ToColorMode(color_mode),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


@torch.no_grad()
def predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Return (softmax probabilities, true labels) for every image in loader."""
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        all_probs.append(torch.softmax(model(images), dim=1).numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    top5 = np.argsort(probs, axis=1)[:, -5:]
    return {
        "top1": float((preds == labels).mean()),
        "top5": float(np.mean([label in row for label, row in zip(labels, top5)])),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "roc_auc": float(
            roc_auc_score(labels, probs[:, 1]) if probs.shape[1] == 2  # binary edge case
            else roc_auc_score(labels, probs, multi_class="ovr", average="macro",
                               labels=list(range(probs.shape[1])))
        ),
    }


def short_name(class_name: str) -> str:
    """'Tomato__Tomato_YellowLeaf__Curl_Virus' -> 'Tomato YellowLeaf Curl Virus'."""
    name = class_name.replace("Pepper__bell", "Pepper").replace("Tomato__Tomato", "Tomato")
    while "__" in name:
        name = name.replace("__", "_")
    return name.replace("_", " ").strip()


def plot_confusion_matrix(labels, preds, class_names, title, path) -> None:
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    ticks = range(len(class_names))
    names = [short_name(c) for c in class_names]
    ax.set_xticks(ticks, names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(ticks, names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(history_path: Path, out_path: Path) -> None:
    data = json.loads(history_path.read_text())
    hist = data["history"]
    x = range(1, len(hist) + 1)
    head_epochs = sum(1 for h in hist if h["phase"] == "head")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, label in ((ax_loss, "loss", "Loss"), (ax_acc, "acc", "Accuracy")):
        ax.plot(x, [h[f"train_{key}"] for h in hist], marker="o", label="train")
        ax.plot(x, [h[f"val_{key}"] for h in hist], marker="o", label="validation")
        ax.axvline(head_epochs + 0.5, color="gray", linestyle="--", linewidth=1)
        ax.text(head_epochs + 0.6, ax.get_ylim()[1], " fine-tuning starts", va="top", fontsize=8, color="gray")
        ax.set_xlabel("Epoch (both phases)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Training curves: {data['run']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    checkpoints = sorted(p for p in MODELS_DIR.glob("*.pt") if not p.stem.endswith("_quick"))
    if not checkpoints:
        raise SystemExit(f"No trained models found in {MODELS_DIR}")

    rows, cache = [], {}
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        run = ckpt_path.stem
        model = build_model(ckpt["model_name"], len(ckpt["class_names"]), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])

        clean_ds = datasets.ImageFolder(DEFAULT_DATA_DIR / "test", eval_transform(ckpt["color_mode"]))
        if clean_ds.classes != ckpt["class_names"]:
            raise SystemExit(f"Class mismatch between test data and {run}")
        noise = AddRGBNoise(TEST_NOISE_STD, NOISE_SEED)
        noisy_ds = datasets.ImageFolder(DEFAULT_DATA_DIR / "test", eval_transform(ckpt["color_mode"], noise))

        probs, labels = predict(model, DataLoader(clean_ds, batch_size=32))
        noise.reset(NOISE_SEED)  # identical noisy images for every model
        noisy_probs, noisy_labels = predict(model, DataLoader(noisy_ds, batch_size=32))

        clean = compute_metrics(probs, labels)
        noisy = compute_metrics(noisy_probs, noisy_labels)
        cache[run] = (probs, labels, ckpt["class_names"])
        rows.append({
            "run": run,
            "model": ckpt["model_name"],
            "color_mode": ckpt["color_mode"],
            "trained_with_noise": ckpt["noise"],
            "val_acc": round(ckpt["best_val_acc"], 4),
            "val_loss": round(ckpt["best_val_loss"], 4),
            "test_top1": round(clean["top1"], 4),
            "test_top5": round(clean["top5"], 4),
            "test_macro_f1": round(clean["macro_f1"], 4),
            "test_roc_auc": round(clean["roc_auc"], 4),
            "noisy_test_top1": round(noisy["top1"], 4),
            "noisy_test_macro_f1": round(noisy["macro_f1"], 4),
        })
        print(f"{run:<24} val {ckpt['best_val_acc']:.3f} | test top1 {clean['top1']:.3f} "
              f"top5 {clean['top5']:.3f} F1 {clean['macro_f1']:.3f} AUC {clean['roc_auc']:.3f} "
              f"| noisy top1 {noisy['top1']:.3f}")

    # Model selection uses VALIDATION only (highest val acc, then lowest val loss).
    best = max(rows, key=lambda r: (r["val_acc"], -r["val_loss"]))
    run = best["run"]
    probs, labels, class_names = cache[run]
    preds = probs.argmax(axis=1)

    plot_confusion_matrix(labels, preds, class_names, f"Confusion matrix (test set): {run}",
                          RESULTS_DIR / f"confusion_matrix_{run}.png")
    report = classification_report(labels, preds, target_names=[short_name(c) for c in class_names], digits=3)
    (RESULTS_DIR / f"classification_report_{run}.txt").write_text(report)

    for hist_path in sorted(RESULTS_DIR.glob("history_*.json")):
        if not hist_path.stem.endswith("_quick"):
            plot_training_curves(hist_path, RESULTS_DIR / f"curves_{hist_path.stem.removeprefix('history_')}.png")

    rows.sort(key=lambda r: r["val_acc"], reverse=True)
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731
    lines = [
        "| Model | Colour | Noise aug. | Val acc | Test Top-1 | Test Top-5 | Test macro-F1 | Test ROC-AUC | Noisy-test Top-1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['color_mode'].upper()} | {'yes' if r['trained_with_noise'] else 'no'} | "
                     f"{pct(r['val_acc'])} | {pct(r['test_top1'])} | {pct(r['test_top5'])} | "
                     f"{r['test_macro_f1']:.3f} | {r['test_roc_auc']:.3f} | {pct(r['noisy_test_top1'])} |")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"\nSelected by validation accuracy: {run}")
    print(f"  test Top-1 {pct(best['test_top1'])} | Top-5 {pct(best['test_top5'])} | "
          f"macro-F1 {best['test_macro_f1']:.3f} | ROC-AUC {best['test_roc_auc']:.3f}")
    print(f"\nPer-class report ({run}):\n{report}")
    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
