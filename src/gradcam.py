"""
Grad-CAM explainability: show WHERE on the leaf the model looked to make its decision.

How Grad-CAM works (interview version):
1. Run the image forward and take the feature maps of the last convolutional stage
   (ResNet50: layer4, 2048 maps of 7x7; DenseNet121: norm5, 1024 maps of 7x7).
2. Backpropagate the score of the predicted class to those feature maps.
3. Average each map's gradient -> one importance weight per map
   ("how much does this map push the score up?").
4. Weighted sum of the maps, then ReLU (keep only positive evidence) -> a 7x7 heatmap.
5. Upsample to 224x224 and overlay on the image. Red = most influential region.

This module is also imported by the Streamlit app (load_checkpoint, preprocess,
predict, gradcam_overlay).

Usage (from the project folder):
    python src/gradcam.py                                  # uses models/resnet50_rgb.pt
    python src/gradcam.py --checkpoint models/densenet121_rgb.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402
from torchvision import transforms  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import DEFAULT_DATA_DIR, IMAGE_SIZE, build_transforms  # noqa: E402
from models import build_model, gradcam_target_layer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "resnet50_rgb.pt"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def short_name(class_name: str) -> str:
    name = class_name.replace("Pepper__bell", "Pepper").replace("Tomato__Tomato", "Tomato")
    while "__" in name:
        name = name.replace("__", "_")
    return name.replace("_", " ").strip()


def load_checkpoint(path: Path):
    """Return (model in eval mode, checkpoint metadata dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def preprocess(img: Image.Image, color_mode: str) -> tuple[torch.Tensor, np.ndarray]:
    """Return (normalized 1x3x224x224 tensor for the model, 224x224 RGB float image for display).

    The model tensor uses EXACTLY the same eval pipeline as training/evaluation
    (dataset.build_transforms), so there is no training-serving skew. An earlier
    version resized with PIL's Image.resize (different interpolation), which
    flipped 2 borderline test predictions.
    """
    img = img.convert("RGB")
    tensor = build_transforms(train=False, color_mode=color_mode)(img).unsqueeze(0)
    display = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(img)  # same resize, RGB for display
    return tensor, np.asarray(display, dtype=np.float32) / 255.0


@torch.no_grad()
def predict(model, tensor: torch.Tensor) -> np.ndarray:
    """Softmax probabilities for one image."""
    return torch.softmax(model(tensor), dim=1)[0].numpy()


def gradcam_overlay(model, model_name: str, tensor: torch.Tensor, rgb_float: np.ndarray, class_idx: int) -> np.ndarray:
    """Heatmap for `class_idx` blended over the image (uint8 RGB, 224x224)."""
    cam = GradCAM(model=model, target_layers=[gradcam_target_layer(model, model_name)])
    heatmap = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(class_idx)])[0]
    return show_cam_on_image(rgb_float, heatmap, use_rgb=True)


def collect_test_images(data_dir: Path, class_names: list[str]) -> list[tuple[Path, int]]:
    items = []
    for idx, name in enumerate(class_names):
        folder = data_dir / "test" / name
        items += [(p, idx) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    return items


def plot_grid(examples, class_names, title, path, cols=5):
    """examples: list of (rgb_float, overlay, true_idx, pred_idx, confidence)."""
    rows = int(np.ceil(len(examples) / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 6.2))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for i, (rgb, overlay, true_idx, pred_idx, conf) in enumerate(examples):
        r, c = (i // cols) * 2, i % cols
        ok = true_idx == pred_idx
        axes[r, c].imshow(rgb)
        axes[r, c].set_title(f"True: {short_name(class_names[true_idx])}", fontsize=8)
        axes[r + 1, c].imshow(overlay)
        axes[r + 1, c].set_title(f"Pred: {short_name(class_names[pred_idx])} ({conf:.0%})", fontsize=8,
                                 color="green" if ok else "red")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps on test images.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint)
    class_names, model_name, color_mode = ckpt["class_names"], ckpt["model_name"], ckpt["color_mode"]
    run = args.checkpoint.stem
    RESULTS_DIR.mkdir(exist_ok=True)

    per_class_done, correct_examples, error_examples = set(), [], []
    for path, true_idx in collect_test_images(args.data_dir, class_names):
        tensor, rgb = preprocess(Image.open(path), color_mode)
        probs = predict(model, tensor)
        pred_idx = int(probs.argmax())
        is_error = pred_idx != true_idx
        wants_correct = not is_error and true_idx not in per_class_done
        if not (is_error or wants_correct):
            continue
        overlay = gradcam_overlay(model, model_name, tensor, rgb, pred_idx)
        example = (rgb, overlay, true_idx, pred_idx, float(probs[pred_idx]))
        if is_error:
            error_examples.append(example)
        else:
            correct_examples.append(example)
            per_class_done.add(true_idx)

    grid_path = RESULTS_DIR / f"gradcam_correct_{run}.png"
    plot_grid(correct_examples, class_names,
              f"Grad-CAM, one correct test image per class ({run})", grid_path)
    print(f"Saved {len(correct_examples)} correct examples -> {grid_path}")

    if error_examples:
        errors_path = RESULTS_DIR / f"gradcam_errors_{run}.png"
        plot_grid(error_examples, class_names,
                  f"Grad-CAM on misclassified test images ({run})", errors_path)
        print(f"Saved {len(error_examples)} misclassified examples -> {errors_path}")
        print("\nMisclassifications (true -> predicted, confidence):")
        for _, _, t, p, conf in error_examples:
            print(f"  {short_name(class_names[t]):<40} -> {short_name(class_names[p]):<40} {conf:.0%}")
    else:
        print("No misclassified test images.")


if __name__ == "__main__":
    main()
