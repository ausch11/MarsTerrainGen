"""Frozen DINOv2 target encoder, used only to build training supervision caches."""
from PIL import Image
import torch
import torch.nn.functional as F

from .utils import model_identity


class DinoTeacher:
    def __init__(self, cfg, device):
        from transformers import AutoImageProcessor, AutoModel
        self.cfg, self.device = cfg, device
        self.processor = AutoImageProcessor.from_pretrained(
            cfg.teacher_model_name_or_path, revision=cfg.teacher_revision,
            local_files_only=cfg.local_files_only, use_fast=False)
        self.model = AutoModel.from_pretrained(
            cfg.teacher_model_name_or_path, revision=cfg.teacher_revision,
            local_files_only=cfg.local_files_only).to(device, dtype=torch.float32)
        if self.model.config.model_type not in {"dinov2", "dinov2_with_registers"}:
            raise ValueError("B1 currently supports DINOv2 and DINOv2-with-registers teachers")
        self.model.requires_grad_(False).eval()
        self.hidden_size = self.model.config.hidden_size
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        if cfg.teacher_input_size % self.model.config.patch_size:
            raise ValueError("teacher_input_size must be divisible by teacher patch_size")
        self.identity = model_identity(cfg.teacher_model_name_or_path, cfg.teacher_revision,
                                       model_config=self.model.config.to_dict())
        self.identity.update(processor=self.processor.to_dict(),
                             input_size=cfg.teacher_input_size, pooling=cfg.teacher_pooling,
                             hidden_size=self.hidden_size, dtype="float32")

    @torch.no_grad()
    def encode(self, images):
        # Preserve the full VAE crop, rather than silently center-cropping it again.
        resized = [im.resize((self.cfg.teacher_input_size,) * 2, Image.Resampling.BICUBIC)
                   for im in images]
        inputs = self.processor(images=resized, do_resize=False, do_center_crop=False,
                                return_tensors="pt")
        tokens = self.model(pixel_values=inputs.pixel_values.to(self.device)).last_hidden_state
        if self.cfg.teacher_pooling == "cls":
            features = tokens[:, 0]
        else:
            features = tokens[:, 1 + self.num_register_tokens:].mean(dim=1)
        return F.normalize(features.float(), dim=-1).cpu()

