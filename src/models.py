"""
Model construction and freeze/unfreeze helpers for transfer learning.

Transfer learning idea: ResNet50 and DenseNet121 were pretrained on ImageNet
(1.2M images, 1000 classes), so their early layers already detect general
features (edges, textures, shapes). We replace only the final classification
layer with a new one for our 15 leaf classes, then:
  Phase 1: train ONLY the new head (backbone frozen)
  Phase 2: also unfreeze the LAST block and fine-tune it with a small learning rate
"""

import torch.nn as nn
from torchvision import models

MODEL_NAMES = ("resnet50", "densenet121")


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Load a pretrained backbone and replace its classifier with a `num_classes` head."""
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # 2048 -> 15
    elif name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)  # 1024 -> 15
    else:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    return model


def head(model: nn.Module, name: str) -> nn.Module:
    """The new classification layer we added."""
    return model.fc if name == "resnet50" else model.classifier


def last_block(model: nn.Module, name: str) -> list[nn.Module]:
    """The deepest feature block, which is unfrozen during fine-tuning."""
    if name == "resnet50":
        return [model.layer4]
    return [model.features.denseblock4, model.features.norm5]


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional stage: its feature maps are what Grad-CAM visualizes."""
    if name == "resnet50":
        return model.layer4[-1]
    return model.features.norm5


def freeze_backbone(model: nn.Module, name: str) -> None:
    """Phase 1: freeze everything except the new head."""
    for p in model.parameters():
        p.requires_grad = False
    for p in head(model, name).parameters():
        p.requires_grad = True


def unfreeze_last_block(model: nn.Module, name: str) -> None:
    """Phase 2: additionally unfreeze the last feature block."""
    for module in last_block(model, name):
        for p in module.parameters():
            p.requires_grad = True


def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Keep BatchNorm layers whose weights are frozen in eval mode.

    Calling model.train() puts EVERY BatchNorm layer in training mode, which makes
    it keep updating its running mean/variance from our small batches, even when
    its weights are frozen. That silently drifts the pretrained statistics, so we
    switch those layers back to eval mode after each model.train() call.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
