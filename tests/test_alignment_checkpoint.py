from dataclasses import replace
from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from mars_lora.alignment import alignment_loss
from mars_lora.checkpoint import load_bundle_weights, save_bundle
from mars_lora.config import TrainConfig
from mars_lora.losses import per_sample_diffusion_loss
from mars_lora.models import TrainingModel, add_lora


@pytest.mark.parametrize("checkpointing", [False, True])
def test_alignment_alone_backpropagates_into_lora(tiny_models, checkpointing):
    unet, text = tiny_models
    cfg = TrainConfig(unet_rank=2, text_encoder_rank=2, projector_hidden_dim=32,
                      mixed_precision="no", max_token_length=16)
    add_lora(unet, text, cfg)
    if checkpointing:
        unet.enable_gradient_checkpointing(gradient_checkpointing_func=partial(checkpoint, use_reentrant=False))
    model = TrainingModel(unet, text, cfg, teacher_dim=24).train()
    target = torch.randn(2, 24, requires_grad=True)
    _, projected = model(torch.randn(2, 4, 16, 16), torch.tensor([3, 8]),
                         torch.randint(0, 16, (2, 16)))
    alignment_loss(projected, target).mean().backward()  # NO diffusion loss in this test.
    assert target.grad is None
    lora_grad = sum(p.grad.abs().sum().item() for n, p in unet.named_parameters()
                    if "lora_" in n and p.grad is not None)
    assert lora_grad > 0
    assert sum(p.grad.abs().sum().item() for p in model.projector.parameters()) > 0
    assert all(p.grad is None for p in unet.parameters() if not p.requires_grad)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    assert model.tap.feature is None and not model.tap.active


def test_baseline_has_no_projector_or_feature_dependency(tiny_models):
    unet, text = tiny_models
    cfg = TrainConfig(b1_enabled=False, unet_rank=2, text_encoder_rank=2)
    add_lora(unet, text, cfg)
    model = TrainingModel(unet, text, cfg)
    output, projected = model(torch.randn(1, 4, 16, 16), torch.tensor([5]), torch.ones(1, 16, dtype=torch.long))
    assert projected is None and model.projector is None and output.shape == (1, 4, 16, 16)


def test_bundle_restores_projector_and_nonunit_alpha(tmp_path, tiny_models):
    from diffusers import DDPMScheduler
    unet, text = tiny_models
    cfg = TrainConfig(unet_rank=2, unet_alpha=7, text_encoder_rank=2,
                      text_encoder_alpha=3, projector_hidden_dim=32)
    add_lora(unet, text, cfg)
    model = TrainingModel(unet, text, cfg, teacher_dim=24).eval()
    for p in model.parameters():
        if p.requires_grad:
            p.data.add_(torch.randn_like(p) * 0.01)
    inputs = (torch.randn(1, 4, 16, 16), torch.tensor([3]), torch.ones(1, 16, dtype=torch.long))
    with torch.no_grad():
        before = model(*inputs)
    save_bundle(model, cfg, DDPMScheduler(), tmp_path, include_projector=True)
    for p in model.parameters():
        if p.requires_grad:
            p.data.zero_()
    load_bundle_weights(model, tmp_path, include_projector=True)
    with torch.no_grad():
        after = model(*inputs)
    torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)
    assert model.unet.peft_config["default"].lora_alpha == 7


@pytest.mark.parametrize("kind", ["epsilon", "v_prediction"])
def test_min_snr_prediction_specific_weight(kind):
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler(num_train_timesteps=20, prediction_type=kind)
    times = torch.tensor([1, 10])
    result = per_sample_diffusion_loss(torch.ones(2, 4, 2, 2), torch.zeros(2, 4, 2, 2), times, scheduler, 5)
    alpha = scheduler.alphas_cumprod[times]
    snr = alpha / (1 - alpha)
    expected = snr.clamp(max=5) / (snr if kind == "epsilon" else snr + 1)
    torch.testing.assert_close(result, expected)

