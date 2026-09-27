import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    torch.set_num_threads(1)


def make_pair(folder, name, seed):
    folder.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(image).save(folder / f"{name}.png")
    (folder / f"{name}.txt").write_text("Mars, ridges, fine texture, dense coverage", encoding="utf-8")


@pytest.fixture
def tiny_models():
    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel
    torch.manual_seed(22)
    unet = UNet2DConditionModel(sample_size=16, in_channels=4, out_channels=4,
        layers_per_block=1, block_out_channels=(16, 32), norm_num_groups=8,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32, attention_head_dim=4)
    text = CLIPTextModel(CLIPTextConfig(vocab_size=16, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=16,
        bos_token_id=0, eos_token_id=1, pad_token_id=1))
    return unet, text


@pytest.fixture
def tiny_workspace(tmp_path):
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import (BitImageProcessor, CLIPTextConfig, CLIPTextModel, CLIPTokenizer,
                              Dinov2Config, Dinov2Model)
    from transformers.models.clip.tokenization_clip import bytes_to_unicode
    from mars_lora.config import TrainConfig
    torch.manual_seed(22)
    base, teacher_path = tmp_path / "sd", tmp_path / "teacher"
    tokenizer_files = tmp_path / "tokenizer_files"
    tokenizer_files.mkdir()
    chars = list(bytes_to_unicode().values())
    tokens = chars + [x + "</w>" for x in chars] + ["<|startoftext|>", "<|endoftext|>"]
    vocab = {token: i for i, token in enumerate(tokens)}
    (tokenizer_files / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tokenizer_files / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = CLIPTokenizer(str(tokenizer_files / "vocab.json"), str(tokenizer_files / "merges.txt"),
                              model_max_length=16)
    text = CLIPTextModel(CLIPTextConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=16,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id))
    unet = UNet2DConditionModel(sample_size=16, in_channels=4, out_channels=4,
        layers_per_block=1, block_out_channels=(16, 32), norm_num_groups=8,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32, attention_head_dim=4)
    vae = AutoencoderKL(block_out_channels=(16, 32), layers_per_block=1, latent_channels=4,
        norm_num_groups=8, sample_size=32, down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"))
    scheduler = DDPMScheduler(num_train_timesteps=20, beta_schedule="linear", clip_sample=False, steps_offset=1)
    StableDiffusionPipeline(vae=vae, text_encoder=text, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=None, feature_extractor=None,
        requires_safety_checker=False).save_pretrained(base)
    teacher = Dinov2Model(Dinov2Config(image_size=32, patch_size=8, hidden_size=32,
        num_hidden_layers=1, num_attention_heads=4, mlp_ratio=2))
    teacher.save_pretrained(teacher_path)
    BitImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}).save_pretrained(teacher_path)
    train_dir, val_dir = tmp_path / "train", tmp_path / "val"
    for cls in range(2):
        for idx in range(3):
            make_pair(train_dir / f"{cls}_class", str(idx), 10 * cls + idx)
        make_pair(val_dir / f"{cls}_class", "held_out", 100 + cls)
    cfg = TrainConfig(pretrained_model_name_or_path=str(base), train_data_dir=str(train_dir),
        validation_data_dir=str(val_dir), output_dir=str(tmp_path / "run"), cache_dir=str(tmp_path / "cache"),
        expected_unique_images=6, expected_num_classes=2, expected_images_per_class=3,
        teacher_model_name_or_path=str(teacher_path), teacher_input_size=32,
        resolution=32, train_batch_size=2, gradient_accumulation_steps=2,
        dataloader_num_workers=0, persistent_data_loader_workers=False,
        num_train_epochs=2, max_train_steps=4, mixed_precision="no", report_to="none",
        unet_rank=2, unet_alpha=4, text_encoder_rank=2, text_encoder_alpha=2,
        projector_hidden_dim=32, alignment_warmup_steps=0, lr_warmup_steps=0,
        unet_lr=1e-3, text_encoder_lr=1e-3, projector_lr=1e-3,
        enable_xformers=False, local_files_only=True, max_token_length=16,
        checkpointing_steps=1, checkpointing_epochs=1, validation_epochs=1,
        validation_loss_batch_size=2, num_validation_images=1, validation_inference_steps=2,
        cache_batch_size=2, log_every=1)
    return cfg, tmp_path

