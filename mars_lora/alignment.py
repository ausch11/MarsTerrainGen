"""B1: target-image global representation supervision (no image conditioning)."""
import torch
from torch import nn
import torch.nn.functional as F


class GlobalProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
                                 nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, feature_map):
        return self.net(feature_map.float().mean(dim=(-2, -1)))


class MidBlockTap:
    """Capture only while explicitly armed; inference never retains training graphs."""
    def __init__(self, mid_block):
        self.active, self.feature = False, None
        self.handle = mid_block.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.active:
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor) or value.ndim != 4:
                raise RuntimeError("Expected a 4D UNet mid-block feature tensor")
            self.feature = value  # NEVER detach: alignment must train the LoRA.

    def begin(self):
        self.feature, self.active = None, True

    def take(self):
        feature = self.feature
        self.feature, self.active = None, False
        if feature is None:
            raise RuntimeError("UNet mid-block hook did not run")
        if torch.is_grad_enabled() and not feature.requires_grad:
            raise RuntimeError("B1 features are detached; check checkpointing/LoRA gradient flow")
        return feature

    def clear(self):
        self.feature, self.active = None, False

    def close(self):
        self.clear()
        self.handle.remove()


def alignment_loss(predicted, target):
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError(f"Alignment feature shape mismatch: {predicted.shape} vs {target.shape}")
    return 1.0 - F.cosine_similarity(predicted.float(), target.detach().float(), dim=-1)


def alignment_coefficient(cfg, global_step):
    if not cfg.b1_enabled:
        return 0.0
    if cfg.alignment_warmup_steps == 0:
        return cfg.alignment_weight
    return cfg.alignment_weight * min(1.0, (global_step + 1) / cfg.alignment_warmup_steps)

