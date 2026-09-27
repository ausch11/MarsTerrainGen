"""Notebook cell 8: diffusion objective plus independent B1 supervision."""
import torch
import torch.nn.functional as F


def diffusion_target(scheduler, latents, noise, timesteps):
    if scheduler.config.prediction_type == "epsilon":
        return noise
    if scheduler.config.prediction_type == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unsupported prediction_type: {scheduler.config.prediction_type}")


def per_sample_diffusion_loss(prediction, target, timesteps, scheduler, snr_gamma=None):
    loss = F.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)
    if snr_gamma is not None:
        alpha = scheduler.alphas_cumprod.to(timesteps.device)[timesteps].float()
        snr = alpha / (1.0 - alpha).clamp_min(1e-8)
        numerator = torch.minimum(snr, torch.full_like(snr, snr_gamma))
        denominator = snr if scheduler.config.prediction_type == "epsilon" else snr + 1.0
        loss = loss * numerator / denominator.clamp_min(1e-8)
    return loss


@torch.no_grad()
def batch_latents(batch, vae, cfg, device, dtype):
    if batch["latents"] is not None:
        return batch["latents"].to(device=device, dtype=dtype)
    vae_dtype = next(vae.parameters()).dtype
    images = batch["pixel_values"].to(device=device, dtype=vae_dtype)
    return (vae.encode(images).latent_dist.sample() * vae.config.scaling_factor).to(dtype)

