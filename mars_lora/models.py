"""Notebook cell 6: frozen SD1.5 backbones, LoRA and B1 training-only projector."""
from functools import partial
import logging

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .alignment import GlobalProjector, MidBlockTap
from .text import encode_chunk_ids

LOG = logging.getLogger(__name__)


def weight_dtype(precision):
    return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def cast_trainable_fp32(model):
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()


def add_lora(unet, text_encoder, cfg):
    from peft import LoraConfig
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=cfg.unet_rank, lora_alpha=cfg.unet_alpha,
        lora_dropout=cfg.lora_dropout, init_lora_weights="gaussian",
        target_modules=cfg.unet_target_modules))
    if cfg.train_text_encoder_lora:
        text_encoder.add_adapter(LoraConfig(r=cfg.text_encoder_rank,
            lora_alpha=cfg.text_encoder_alpha, lora_dropout=cfg.lora_dropout,
            init_lora_weights="gaussian", target_modules=cfg.text_encoder_target_modules))
    cast_trainable_fp32(unet)
    cast_trainable_fp32(text_encoder)


class TrainingModel(nn.Module):
    def __init__(self, unet, text_encoder, cfg, teacher_dim=None):
        super().__init__()
        self.unet, self.text_encoder, self.cfg = unet, text_encoder, cfg
        self.projector = None
        self.tap = None
        if cfg.b1_enabled:
            if not teacher_dim or unet.mid_block is None:
                raise ValueError("B1 requires a teacher feature dimension and UNet mid_block")
            self.projector = GlobalProjector(unet.config.block_out_channels[-1],
                cfg.projector_hidden_dim, teacher_dim).to(next(unet.parameters()).device)
            self.tap = MidBlockTap(unet.mid_block)

    def train(self, mode=True):
        super().train(mode)
        if not self.cfg.train_text_encoder_lora:
            self.text_encoder.eval()
        return self

    def forward(self, noisy_latents, timesteps, input_ids, supervise=True):
        if self.cfg.train_text_encoder_lora:
            encoded = encode_chunk_ids(self.text_encoder, input_ids)
        else:
            with torch.no_grad():
                encoded = encode_chunk_ids(self.text_encoder, input_ids)
        capture = supervise and self.projector is not None
        if capture:
            self.tap.begin()
        try:
            prediction = self.unet(noisy_latents, timesteps, encoded, return_dict=False)[0]
            projected = self.projector(self.tap.take()) if capture else None
            return prediction, projected
        finally:
            if self.tap is not None:
                self.tap.clear()


def load_components(cfg, device):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    common = dict(revision=cfg.revision, local_files_only=cfg.local_files_only)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path,
                                             subfolder="tokenizer", **common)
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path,
                                               subfolder="text_encoder", **common)
    if cfg.max_token_length > text_encoder.config.max_position_embeddings:
        raise ValueError("max_token_length exceeds CLIP positional embedding length")
    tokenizer.model_max_length = cfg.max_token_length
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path,
                                       subfolder="vae", variant=cfg.variant, **common)
    unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path,
                                             subfolder="unet", variant=cfg.variant, **common)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path,
                                                  subfolder="scheduler", **common)
    if cfg.prediction_type:
        noise_scheduler.register_to_config(prediction_type=cfg.prediction_type)
    if noise_scheduler.config.prediction_type not in {"epsilon", "v_prediction"}:
        raise ValueError("Only epsilon and v_prediction are supported")
    dtype = weight_dtype(cfg.mixed_precision)
    unet.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)
    vae.requires_grad_(False).eval().to(device, dtype=getattr(torch, cfg.vae_dtype))
    add_lora(unet, text_encoder, cfg)
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing(
            gradient_checkpointing_func=partial(checkpoint, use_reentrant=False))
        if cfg.train_text_encoder_lora:
            text_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    if cfg.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except (ImportError, ModuleNotFoundError, ValueError, RuntimeError) as exc:
            LOG.warning("xformers unavailable (%s); using the existing attention processor", exc)
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def trainable_summary(model):
    result = {}
    for name, module in (("unet", model.unet), ("text_encoder", model.text_encoder),
                         ("projector", model.projector)):
        parameters = [] if module is None else [p for p in module.parameters() if p.requires_grad]
        result[name] = {"parameters": sum(p.numel() for p in parameters),
                        "dtypes": sorted({str(p.dtype) for p in parameters})}
    return result
