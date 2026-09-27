import math
import torch


def make_optimizer(model, cfg):
    groups = [{"params": [p for p in model.unet.parameters() if p.requires_grad],
               "lr": cfg.unet_lr, "name": "unet"}]
    if cfg.train_text_encoder_lora:
        groups.append({"params": [p for p in model.text_encoder.parameters() if p.requires_grad],
                       "lr": cfg.text_encoder_lr, "name": "text_encoder"})
    if model.projector is not None:
        groups.append({"params": list(model.projector.parameters()), "lr": cfg.projector_lr,
                       "name": "projector"})
    if any(not group["params"] for group in groups):
        raise ValueError("An optimizer group has no trainable parameters")
    if any(p.dtype != torch.float32 for group in groups for p in group["params"]):
        raise ValueError("All trainable parameters must be float32 before optimizer construction")
    cls = torch.optim.AdamW
    if cfg.use_8bit_adam:
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError as exc:
            raise ImportError("Install a compatible bitsandbytes, or disable use_8bit_adam") from exc
        cls = AdamW8bit
    return cls(groups, betas=(cfg.adam_beta1, cfg.adam_beta2),
               weight_decay=cfg.adam_weight_decay, eps=cfg.adam_epsilon)


def make_scheduler(optimizer, cfg, total_steps):
    """One explicit scheduler step per successful optimizer update; no world-size multiplier."""
    def multiplier(step):
        if cfg.lr_scheduler == "constant":
            return 1.0
        if step < cfg.lr_warmup_steps:
            return step / max(1, cfg.lr_warmup_steps)
        if cfg.lr_scheduler == "constant_with_warmup":
            return 1.0
        progress = min(1.0, max(0.0, (step - cfg.lr_warmup_steps) /
                                max(1, total_steps - cfg.lr_warmup_steps)))
        return 1.0 - progress if cfg.lr_scheduler == "linear" else 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

