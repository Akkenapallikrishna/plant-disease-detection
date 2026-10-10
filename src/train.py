"""
Train one experiment: a model (resnet50 / densenet121) with a colour mode (rgb / hsv)
and optional Gaussian-noise augmentation.

Two-phase transfer learning with early stopping:
  Phase 1 "head":     backbone frozen, only the new classifier learns (lr 1e-3)
  Phase 2 "finetune": last block unfrozen too, 10x smaller lr (1e-4)
The checkpoint with the LOWEST validation loss across both phases is saved.

Examples (run from the project folder):
    python src/train.py --model resnet50 --color-mode rgb --quick      # 1-minute smoke test
    python src/train.py --model resnet50 --color-mode rgb
    python src/train.py --model densenet121 --color-mode hsv --noise
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import COLOR_MODES, DEFAULT_DATA_DIR, get_dataloaders  # noqa: E402
from models import (  # noqa: E402
    MODEL_NAMES,
    build_model,
    count_parameters,
    freeze_backbone,
    set_frozen_batchnorm_eval,
    unfreeze_last_block,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_name(model_name: str, color_mode: str, noise: bool) -> str:
    return f"{model_name}_{color_mode}" + ("_noise" if noise else "")


def run_epoch(model, loader, criterion, optimizer=None, max_batches=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    if training:
        model.train()
        set_frozen_batchnorm_eval(model)
    else:
        model.eval()

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for i, (images, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / seen, correct / seen


def train_phase(phase, model, loaders, criterion, lr, epochs, patience, state, history, max_batches):
    """Train the currently-trainable parameters with early stopping on validation loss."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    trainable, total = count_parameters(model)
    print(f"\n=== Phase '{phase}': lr={lr}, trainable params {trainable:,} / {total:,} ===")

    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, max_batches)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, None, max_batches)
        elapsed = time.time() - start

        improved = val_loss < state["best_val_loss"]
        if improved:
            state.update(best_val_loss=val_loss, best_val_acc=val_acc,
                         best_state=copy.deepcopy(model.state_dict()), best_phase=phase, best_epoch=epoch)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append({"phase": phase, "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "seconds": elapsed})
        print(f"  epoch {epoch:>2}  train loss {train_loss:.4f} acc {train_acc:.3f}  |  "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}  |  {elapsed:5.1f}s" + ("  *best" if improved else ""))

        if epochs_without_improvement >= patience:
            print(f"  Early stopping: no val-loss improvement for {patience} epochs.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a plant disease classifier.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="resnet50")
    parser.add_argument("--color-mode", choices=COLOR_MODES, default="rgb")
    parser.add_argument("--noise", action="store_true", help="add Gaussian noise augmentation to training images")
    parser.add_argument("--epochs-head", type=int, default=5)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--quick", action="store_true", help="smoke test: 1 epoch per phase, 3 batches each")
    parser.add_argument("--no-pretrained", action="store_true", help="random weights (testing only)")
    args = parser.parse_args()

    set_seed(args.seed)
    name = run_name(args.model, args.color_mode, args.noise)
    max_batches = 3 if args.quick else None
    epochs_head = 1 if args.quick else args.epochs_head
    epochs_finetune = 1 if args.quick else args.epochs_finetune
    print(f"Run: {name}{'  (QUICK TEST)' if args.quick else ''}  |  torch threads: {torch.get_num_threads()}")

    loaders, class_names = get_dataloaders(args.data_dir, args.color_mode, args.noise,
                                           args.batch_size, seed=args.seed)
    model = build_model(args.model, len(class_names), pretrained=not args.no_pretrained)
    criterion = nn.CrossEntropyLoss()
    state = {"best_val_loss": float("inf")}
    history: list[dict] = []
    start = time.time()

    # Phase 1: head only
    freeze_backbone(model, args.model)
    train_phase("head", model, loaders, criterion, args.lr_head, epochs_head, args.patience,
                state, history, max_batches)

    # Phase 2: continue from the best weights so far, unfreeze the last block
    model.load_state_dict(state["best_state"])
    unfreeze_last_block(model, args.model)
    train_phase("finetune", model, loaders, criterion, args.lr_finetune, epochs_finetune, args.patience,
                state, history, max_batches)

    total_minutes = (time.time() - start) / 60
    suffix = "_quick" if args.quick else ""
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    checkpoint_path = MODELS_DIR / f"{name}{suffix}.pt"
    torch.save({
        "model_name": args.model, "color_mode": args.color_mode, "noise": args.noise,
        "class_names": class_names, "state_dict": state["best_state"],
        "best_val_loss": state["best_val_loss"], "best_val_acc": state["best_val_acc"],
        "best_phase": state["best_phase"], "best_epoch": state["best_epoch"],
    }, checkpoint_path)

    history_path = RESULTS_DIR / f"history_{name}{suffix}.json"
    with open(history_path, "w") as f:
        json.dump({"run": name, "args": {k: str(v) for k, v in vars(args).items()},
                   "minutes": total_minutes, "history": history}, f, indent=2)

    print(f"\nBest: val loss {state['best_val_loss']:.4f}, val acc {state['best_val_acc']:.3f} "
          f"(phase '{state['best_phase']}', epoch {state['best_epoch']})")
    print(f"Total time: {total_minutes:.1f} min")
    print(f"Saved model   -> {checkpoint_path}")
    print(f"Saved history -> {history_path}")


if __name__ == "__main__":
    main()
