import pytest
import torch
import torch.nn as nn

from models import (
    MODEL_NAMES,
    build_model,
    count_parameters,
    freeze_backbone,
    gradcam_target_layer,
    head,
    last_block,
    set_frozen_batchnorm_eval,
    unfreeze_last_block,
)

NUM_CLASSES = 15


@pytest.fixture(params=MODEL_NAMES)
def model_and_name(request):
    torch.manual_seed(0)
    return build_model(request.param, NUM_CLASSES, pretrained=False), request.param


def test_output_has_one_score_per_class(model_and_name):
    model, _ = model_and_name
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, NUM_CLASSES)


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_model("vgg16", NUM_CLASSES, pretrained=False)


def test_phase1_only_head_is_trainable(model_and_name):
    model, name = model_and_name
    freeze_backbone(model, name)
    trainable, total = count_parameters(model)
    head_params = sum(p.numel() for p in head(model, name).parameters())
    assert trainable == head_params
    assert trainable < total


def test_phase2_unfreezes_last_block_only(model_and_name):
    model, name = model_and_name
    freeze_backbone(model, name)
    phase1, total = count_parameters(model)
    unfreeze_last_block(model, name)
    phase2, _ = count_parameters(model)
    block_params = sum(p.numel() for m in last_block(model, name) for p in m.parameters())
    assert phase2 == phase1 + block_params
    assert phase2 < total


def test_frozen_batchnorm_stays_in_eval_mode(model_and_name):
    model, name = model_and_name
    freeze_backbone(model, name)
    unfreeze_last_block(model, name)
    model.train()
    set_frozen_batchnorm_eval(model)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            is_frozen = not any(p.requires_grad for p in module.parameters())
            assert module.training is (not is_frozen)


def test_gradcam_target_layer_is_part_of_model(model_and_name):
    model, name = model_and_name
    target = gradcam_target_layer(model, name)
    assert any(target is m for m in model.modules())
