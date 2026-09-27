"""Notebook cells 6/8/9: training orchestration, baseline/B1 share the same diffusion path."""
import gc
import json
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .alignment import alignment_coefficient, alignment_loss
from .cache import build_cache, make_cache
from .checkpoint import (register_hooks, resolve_resume, run_signature,
                         save_bundle, save_checkpoint)
from .data import (CaptionedImageDataset, EpochSampler, collate_fn, data_summary,
                   discover_images, records_fingerprint, validate_counts)
from .losses import batch_latents, diffusion_target, per_sample_diffusion_loss
from .models import TrainingModel, load_components, trainable_summary, weight_dtype
from .optim import make_optimizer, make_scheduler
from .teacher import DinoTeacher
from .text import audit_caption_lengths
from .utils import atomic_json, environment_info, fingerprint, isolated_rng
from .validation import build_prompts, evaluate_denoising, sample_validation

LOG = logging.getLogger(__name__)


def inspect_data(cfg):
    excludes = [cfg.cache_dir, cfg.output_dir, cfg.validation_data_dir]
    records, missing = discover_images(cfg.train_data_dir, cfg.caption_extension,
                                       cfg.strict_captions, excludes)
    summary = data_summary(records, cfg.repeats)
    summary["missing_captions"] = missing
    validate_counts(summary, cfg)
    validation = []
    if cfg.validation_data_dir:
        if Path(cfg.validation_data_dir).resolve() == Path(cfg.train_data_dir).resolve():
            raise ValueError("Training and validation directories must be different")
        validation, _ = discover_images(cfg.validation_data_dir, cfg.caption_extension, True,
                                         [cfg.cache_dir, cfg.output_dir])
        overlap = {r.image_hash for r in records} & {r.image_hash for r in validation}
        if overlap:
            raise ValueError(f"Train/validation exact image overlap: {len(overlap)} hashes")
    return records, validation, summary


def _append_metrics(path, metrics):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")


def train(cfg, cache_only=False, stop_after_steps=None):
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    cfg.validate()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision, log_with=None if cfg.report_to == "none" else cfg.report_to,
        project_config=ProjectConfiguration(project_dir=cfg.output_dir,
                                           logging_dir=str(Path(cfg.output_dir) / "logs")))
    # The requested runtime is a single 48 GB GPU. Fail explicitly instead of pretending
    # that untested distributed checkpoint/dataloader behavior is supported.
    if accelerator.num_processes != 1:
        raise ValueError("This release supports one training process/GPU; use --num_processes 1")
    if accelerator.device.type == "cpu" and cfg.mixed_precision != "no":
        raise ValueError("CPU smoke tests require mixed_precision='no'")
    if cfg.mixed_precision == "bf16" and accelerator.device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is not supported by this CUDA device")
    if cfg.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    records, validation_records, summary = inspect_data(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resume = resolve_resume(output, cfg.resume_from_checkpoint)
    if not cache_only and resume is None and (output / "run.json").exists():
        raise FileExistsError("Output contains an existing run. Choose a new output_dir or explicitly resume.")
    set_seed(cfg.seed)
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_components(cfg, accelerator.device)
    text_summary = audit_caption_lengths(records + validation_records, tokenizer,
                                         cfg.max_token_length, cfg.max_prompt_chunks)
    LOG.info("Caption token audit: %s", text_summary)
    # Cache construction and pretrained teacher initialization do not perturb training RNG.
    with isolated_rng():
        teacher = DinoTeacher(cfg, accelerator.device) if cfg.b1_enabled else None
        teacher_dim = teacher.hidden_size if teacher else None
        cache = make_cache(cfg, vae, teacher)
        build_cache(records, cfg, cache, vae, teacher, accelerator.device)
        if validation_records and cfg.cache_latents:
            build_cache(validation_records, cfg, cache, vae, teacher, accelerator.device, kinds=["latent"])
        del teacher
    gc.collect()
    if cache_only:
        accelerator.end_training()
        return {"cache_dir": str(cache.root), "signatures": cache.signatures}
    if cfg.cache_latents:
        vae.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dataset = CaptionedImageDataset(records, tokenizer, cfg, cache, training=True)
    sampler = EpochSampler(dataset, cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.train_batch_size, sampler=sampler,
        collate_fn=collate_fn, num_workers=cfg.dataloader_num_workers,
        persistent_workers=cfg.persistent_data_loader_workers and cfg.dataloader_num_workers > 0,
        pin_memory=accelerator.device.type == "cuda",
        generator=torch.Generator().manual_seed(cfg.seed + 100))
    validation_loader = None
    if validation_records:
        validation_loader = DataLoader(
            CaptionedImageDataset(validation_records, tokenizer, cfg, cache, training=False),
            batch_size=cfg.validation_loss_batch_size, shuffle=False, collate_fn=collate_fn,
            num_workers=0, generator=torch.Generator().manual_seed(cfg.validation_seed))
    else:
        LOG.warning("No held-out validation data: best model selection is disabled; previews are not a test set.")
    prompts = build_prompts(records, cfg)
    atomic_json(output / "validation_prompts.json", prompts)
    model = TrainingModel(unet, text_encoder, cfg, teacher_dim)
    optimizer = make_optimizer(model, cfg)
    raw_optimizer = optimizer
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    updates_per_epoch = math.ceil(len(loader) / cfg.gradient_accumulation_steps)
    total_steps = cfg.max_train_steps or cfg.num_train_epochs * updates_per_epoch
    epochs = math.ceil(total_steps / updates_per_epoch)
    scheduler = make_scheduler(raw_optimizer, cfg, total_steps)
    accelerator.register_for_checkpointing(scheduler)
    register_hooks(accelerator, model, cfg, noise_scheduler)
    signature = run_signature(cfg, fingerprint({"train": records_fingerprint(records),
                              "validation": records_fingerprint(validation_records)}), cache.specs)
    raw_model = accelerator.unwrap_model(model)
    counts = trainable_summary(raw_model)
    LOG.info("Trainable parameters: %s", counts)
    run_metadata = {"config": cfg.to_dict(), "environment": environment_info(),
        "text": text_summary,
        "data": summary, "cache_signatures": cache.signatures, "signature": signature,
        "trainable": counts, "updates_per_epoch": updates_per_epoch, "total_steps": total_steps,
        "effective_full_batch": cfg.train_batch_size * cfg.gradient_accumulation_steps,
        "device": str(accelerator.device)}
    state = {"global_step": 0, "next_epoch": 0, "next_batch": 0, "best_validation_loss": None,
             "signature": signature, "batches_per_epoch": len(loader), "total_steps": total_steps}
    if resume:
        saved = json.loads((resume / "trainer_state.json").read_text(encoding="utf-8"))
        if saved["signature"] != signature or saved["batches_per_epoch"] != len(loader):
            raise ValueError("Resume recipe/data/cache differs from checkpoint; use the original recipe or start a new run")
        accelerator.load_state(str(resume))
        state = saved
        LOG.info("Resumed step %s, next epoch %s, next batch %s", state["global_step"],
                 state["next_epoch"], state["next_batch"])
    else:
        # B0/B1 get identical diffusion RNG regardless of projector/cache initialization.
        set_seed(cfg.seed + 1)
    atomic_json(output / "run.json", run_metadata)
    accelerator.init_trackers("mars-lora-b1", config={
        "b1_enabled": cfg.b1_enabled, "total_steps": total_steps,
        "train_batch_size": cfg.train_batch_size, "seed": cfg.seed})
    dtype = weight_dtype(cfg.mixed_precision)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    start_epoch = state["next_epoch"]
    interrupted = False
    progress = tqdm(total=total_steps, initial=state["global_step"], desc="optimizer updates")
    try:
        for epoch in range(start_epoch, epochs):
            if state["global_step"] >= total_steps:
                break
            model.train()
            sampler.set_epoch(epoch)
            # Accelerate DataLoaderShard has its own epoch counter; keep both in sync on resume.
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch)
            skip = state["next_batch"] if epoch == start_epoch else 0
            active_loader = accelerator.skip_first_batches(loader, skip) if skip else loader
            group_sums = {"diffusion": 0.0, "alignment": 0.0, "total": 0.0}
            group_count = 0
            epoch_diff_sum, epoch_count = 0.0, 0
            for batch_index, batch in enumerate(active_loader, start=skip):
                coefficient = alignment_coefficient(cfg, state["global_step"])
                with accelerator.accumulate(model):
                    latents = batch_latents(batch, vae, cfg, accelerator.device, dtype)
                    noise = torch.randn_like(latents)
                    if cfg.noise_offset:
                        noise += cfg.noise_offset * torch.randn(
                            (len(latents), latents.shape[1], 1, 1), device=latents.device, dtype=dtype)
                    timesteps = torch.randint(noise_scheduler.config.num_train_timesteps,
                                               (len(latents),), device=latents.device)
                    noisy = noise_scheduler.add_noise(latents, noise, timesteps)
                    prediction, projected = model(noisy, timesteps, batch["input_ids"], supervise=cfg.b1_enabled)
                    target = diffusion_target(noise_scheduler, latents, noise, timesteps)
                    diffusion_values = per_sample_diffusion_loss(prediction, target, timesteps,
                                                                noise_scheduler, cfg.snr_gamma)
                    alignment_values = (alignment_loss(projected, batch["teacher_features"])
                                        if cfg.b1_enabled else torch.zeros_like(diffusion_values))
                    loss = (diffusion_values + coefficient * alignment_values).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite training loss at step {state['global_step']}")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(parameters, cfg.max_grad_norm)
                    optimizer.step()
                    if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                n = len(latents)
                group_sums["diffusion"] += diffusion_values.detach().sum().item()
                group_sums["alignment"] += alignment_values.detach().sum().item()
                group_sums["total"] += loss.detach().item() * n
                group_count += n
                epoch_diff_sum += diffusion_values.detach().sum().item()
                epoch_count += n
                state["next_epoch"] = epoch + int(batch_index + 1 == len(loader))
                state["next_batch"] = 0 if batch_index + 1 == len(loader) else batch_index + 1
                if accelerator.sync_gradients:
                    state["global_step"] += 1
                    step = state["global_step"]
                    metrics = {f"train/{k}_loss": v / group_count for k, v in group_sums.items()}
                    metrics.update({"train/alignment_coefficient": coefficient,
                                    "train/optimizer_step_skipped": int(accelerator.optimizer_step_was_skipped)})
                    metrics.update({f"lr/{g['name']}": g["lr"] for g in optimizer.param_groups})
                    _append_metrics(output / "metrics.jsonl", {"global_step": step, "epoch": epoch + 1, **metrics})
                    if step % cfg.log_every == 0 or step == 1:
                        accelerator.log(metrics, step=step)
                    progress.update(1)
                    progress.set_postfix(diff=f"{metrics['train/diffusion_loss']:.4f}",
                                         align=f"{metrics['train/alignment_loss']:.4f}")
                    group_sums = dict.fromkeys(group_sums, 0.0)
                    group_count = 0
                    if cfg.checkpointing_steps and step % cfg.checkpointing_steps == 0:
                        save_checkpoint(accelerator, cfg, state)
                    if stop_after_steps is not None and step >= stop_after_steps and step < total_steps:
                        save_checkpoint(accelerator, cfg, state)
                        interrupted = True
                        break
                    if step >= total_steps:
                        break
            if interrupted:
                break
            if epoch_count:
                accelerator.log({"train/epoch_diffusion_loss": epoch_diff_sum / epoch_count},
                                step=state["global_step"])
            evaluate_now = (epoch + 1) % cfg.validation_epochs == 0 or state["global_step"] >= total_steps
            if evaluate_now:
                if validation_loader is not None:
                    with isolated_rng():
                        val_loss = evaluate_denoising(model, validation_loader, vae, noise_scheduler, cfg, accelerator)
                    accelerator.log({"validation/denoising_loss": val_loss}, step=state["global_step"])
                    _append_metrics(output / "metrics.jsonl", {"global_step": state["global_step"],
                                                               "validation/denoising_loss": val_loss})
                    if state["best_validation_loss"] is None or val_loss < state["best_validation_loss"]:
                        state["best_validation_loss"] = val_loss
                        save_bundle(raw_model, cfg, noise_scheduler, output / "best-validation")
                        atomic_json(output / "best-validation" / "selection.json", {
                            "metric": "held-out fixed-seed denoising MSE", "value": val_loss,
                            "global_step": state["global_step"], "note": "Not a claim of best FID"})
                sample_validation(raw_model, vae, tokenizer, noise_scheduler, prompts, cfg,
                                  accelerator, state["global_step"])
            if (epoch + 1) % cfg.checkpointing_epochs == 0:
                save_checkpoint(accelerator, cfg, state)
        if not interrupted:
            save_checkpoint(accelerator, cfg, state)
            save_bundle(raw_model, cfg, noise_scheduler, output / "final")
            atomic_json(output / "final" / "completion.json", state)
    finally:
        progress.close()
        accelerator.end_training()
    return {**state, "interrupted": interrupted, "output_dir": str(output)}
