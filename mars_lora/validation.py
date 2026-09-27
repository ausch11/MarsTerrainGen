"""Notebook cell 4: fixed-seed held-out denoising loss and per-class text-only previews."""
import json
from pathlib import Path

import torch

from .losses import diffusion_target, per_sample_diffusion_loss
from .models import weight_dtype
from .utils import atomic_json, isolated_rng


def build_prompts(records, cfg):
    if cfg.validation_prompts_file:
        data = json.loads(Path(cfg.validation_prompts_file).read_text(encoding="utf-8"))
        prompts = [{"class_name": str(i), "prompt": item} if isinstance(item, str) else item
                   for i, item in enumerate(data)]
        if any(not x.get("prompt", "").strip() for x in prompts):
            raise ValueError("Empty validation prompt")
        return prompts
    # A reproducible starting prompt for every actual class, not a made-up list of 17 names.
    prompts = {}
    for record in records:
        prompts.setdefault(record.class_name, record.caption.read_text(encoding="utf-8").strip())
    return [{"class_name": name, "prompt": prompt} for name, prompt in sorted(prompts.items())]


@torch.no_grad()
def evaluate_denoising(model, loader, vae, noise_scheduler, cfg, accelerator):
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.validation_seed)
    dtype = weight_dtype(cfg.mixed_precision)
    try:
        for batch in loader:
            if batch["latents"] is not None:
                latents = batch["latents"].to(accelerator.device, dtype=dtype)
            else:
                images = batch["pixel_values"].to(accelerator.device, dtype=next(vae.parameters()).dtype)
                posterior = vae.encode(images).latent_dist
                latents = posterior.sample(generator=generator) * vae.config.scaling_factor
                latents = latents.to(dtype)
            noise = torch.randn(latents.shape, device=latents.device, dtype=dtype, generator=generator)
            if cfg.noise_offset:
                noise += cfg.noise_offset * torch.randn((len(latents), latents.shape[1], 1, 1),
                    device=latents.device, dtype=dtype, generator=generator)
            timesteps = torch.randint(noise_scheduler.config.num_train_timesteps, (len(latents),),
                                      device=latents.device, generator=generator)
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)
            prediction, _ = model(noisy, timesteps, batch["input_ids"].to(latents.device), supervise=False)
            target = diffusion_target(noise_scheduler, latents, noise, timesteps)
            # Use the same unweighted metric across baseline and B1; never select on total B1 loss.
            values = per_sample_diffusion_loss(prediction, target, timesteps, noise_scheduler)
            total += values.sum().item()
            count += len(values)
    finally:
        model.train(was_training)
    return total / count


@torch.no_grad()
def sample_validation(raw_model, vae, tokenizer, noise_scheduler, prompts, cfg, accelerator, step):
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    if cfg.num_validation_images == 0:
        return
    original_mode = raw_model.training
    vae_device, vae_dtype = next(vae.parameters()).device, next(vae.parameters()).dtype
    raw_model.eval()
    folder = Path(cfg.output_dir) / "validation" / f"step-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with isolated_rng():
            pipe = StableDiffusionPipeline(vae=vae, text_encoder=raw_model.text_encoder,
                tokenizer=tokenizer, unet=raw_model.unet,
                scheduler=DDIMScheduler.from_config(noise_scheduler.config),
                safety_checker=None, feature_extractor=None, requires_safety_checker=False)
            pipe.to(accelerator.device)
            pipe.set_progress_bar_config(disable=True)
            atomic_json(folder / "prompts.json", prompts)
            for class_index, item in enumerate(prompts):
                for image_index in range(cfg.num_validation_images):
                    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.validation_seed + image_index)
                    with accelerator.autocast():
                        image = pipe(item["prompt"], height=cfg.resolution, width=cfg.resolution,
                            num_inference_steps=cfg.validation_inference_steps,
                            guidance_scale=cfg.validation_guidance_scale, generator=generator).images[0]
                    image.save(folder / f"class-{class_index:02d}-seed-{cfg.validation_seed + image_index}.png")
            del pipe
    finally:
        raw_model.train(original_mode)
        vae.to(device=vae_device, dtype=vae_dtype)

