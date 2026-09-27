"""Notebook cell 2: explicit, serializable baseline/B1 configuration."""
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    pretrained_model_name_or_path: str = "/root/autodl-tmp/models_diffuser"
    revision: Optional[str] = None
    variant: Optional[str] = None
    local_files_only: bool = False
    train_data_dir: str = "/root/autodl-tmp/训练数据/all_landform_基于train匹配图像集_qwen3_短语版"
    validation_data_dir: Optional[str] = None
    output_dir: str = "outputs/b1"
    cache_dir: str = "cache"
    caption_extension: str = ".txt"
    strict_captions: bool = True
    repeats: dict = field(default_factory=dict)
    expected_unique_images: Optional[int] = 17000
    expected_num_classes: Optional[int] = 17
    expected_images_per_class: Optional[int] = 1000
    resolution: int = 256
    center_crop: bool = False
    random_flip: bool = False
    cache_latents: bool = True
    cache_batch_size: int = 16
    cache_tag: str = "v1"
    train_batch_size: int = 80
    gradient_accumulation_steps: int = 4
    dataloader_num_workers: int = 8
    persistent_data_loader_workers: bool = True
    num_train_epochs: int = 70
    max_train_steps: Optional[int] = None
    gradient_checkpointing: bool = False
    unet_lr: float = 4e-4
    text_encoder_lr: float = 1e-4
    projector_lr: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    use_8bit_adam: bool = False
    max_grad_norm: float = 1.0
    unet_rank: int = 64
    unet_alpha: int = 64
    text_encoder_rank: int = 32
    text_encoder_alpha: int = 32
    lora_dropout: float = 0.0
    train_text_encoder_lora: bool = True
    unet_target_modules: list = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"])
    text_encoder_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
    mixed_precision: str = "bf16"
    vae_dtype: str = "float32"
    report_to: str = "tensorboard"
    seed: int = 1337
    max_token_length: int = 77
    snr_gamma: Optional[float] = None
    noise_offset: float = 0.0
    prediction_type: Optional[str] = None
    shuffle_caption: bool = True
    keep_tokens: int = 2
    caption_dropout: float = 0.0
    enable_xformers: bool = True
    allow_tf32: bool = True
    b1_enabled: bool = True
    teacher_model_name_or_path: str = "facebook/dinov2-base"
    teacher_revision: Optional[str] = None
    teacher_input_size: int = 224
    teacher_pooling: str = "cls"
    alignment_weight: float = 0.05
    alignment_warmup_steps: int = 378
    projector_hidden_dim: int = 1024
    checkpointing_epochs: int = 5
    checkpointing_steps: Optional[int] = None
    checkpoints_total_limit: int = 5
    resume_from_checkpoint: Optional[str] = None
    validation_epochs: int = 10
    validation_prompts_file: Optional[str] = None
    num_validation_images: int = 4
    validation_inference_steps: int = 30
    validation_guidance_scale: float = 7.5
    validation_loss_batch_size: int = 16
    validation_seed: int = 2026
    log_every: int = 10

    def validate(self):
        for name in ("resolution", "train_batch_size", "gradient_accumulation_steps",
                     "num_train_epochs", "cache_batch_size", "checkpointing_epochs",
                     "checkpoints_total_limit", "validation_epochs", "max_token_length",
                     "projector_hidden_dim", "teacher_input_size", "validation_loss_batch_size",
                     "unet_rank", "text_encoder_rank", "log_every", "validation_inference_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.resolution % 8:
            raise ValueError("resolution must be divisible by 8")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be no/fp16/bf16")
        if self.vae_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Invalid vae_dtype")
        if self.random_flip:
            raise ValueError("This cache-aligned B1 implementation requires random_flip=false")
        if self.lr_scheduler not in {"cosine", "constant", "constant_with_warmup", "linear"}:
            raise ValueError("Unsupported lr_scheduler")
        if self.report_to not in {"none", "tensorboard"}:
            raise ValueError("report_to must be none/tensorboard")
        if self.teacher_pooling not in {"cls", "patch_mean"}:
            raise ValueError("teacher_pooling must be cls/patch_mean")
        if self.prediction_type not in {None, "epsilon", "v_prediction"}:
            raise ValueError("Unsupported prediction_type")
        if not 0 <= self.caption_dropout <= 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("Invalid dropout probability")
        if any(not isinstance(v, int) or v < 1 for v in self.repeats.values()):
            raise ValueError("repeats values must be positive integers; numeric class names are not repeats")
        for name in ("max_train_steps", "checkpointing_steps", "snr_gamma"):
            if getattr(self, name) is not None and getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if min(self.alignment_weight, self.alignment_warmup_steps, self.lr_warmup_steps,
               self.dataloader_num_workers, self.keep_tokens, self.noise_offset,
               self.num_validation_images) < 0:
            raise ValueError("Weights, counts and warmup values cannot be negative")
        if min(self.unet_lr, self.text_encoder_lr, self.projector_lr) <= 0:
            raise ValueError("Learning rates must be positive")
        if self.resume_from_checkpoint == "None":
            raise ValueError('Use JSON null, not the string "None"')
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path, overrides=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.update(overrides or {})
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return TrainConfig(**data).validate()

