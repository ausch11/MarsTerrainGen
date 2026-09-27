"""Text-only export loader. Neither DINOv2 nor the B1 projector is needed."""
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from .checkpoint import restore_adapter
from .models import weight_dtype
from .text import chunk_prompt, encode_prompt_embeds, prompt_token_ids
from .utils import atomic_json


def read_prompt_directory(prompt_dir, recursive=False, one_prompt_per_line=False):
    """Read UTF-8 text files deterministically; text is data, never executable instructions."""
    root = Path(prompt_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Prompt directory does not exist: {root}")
    candidates = root.rglob("*") if recursive else root.iterdir()
    files = sorted((p for p in candidates if p.is_file() and p.suffix.lower() == ".txt"),
                   key=lambda p: p.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"No .txt prompt files found in {root} (recursive={recursive})")
    records = []
    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        prompts = ([line.strip() for line in text.splitlines() if line.strip()]
                   if one_prompt_per_line else [" ".join(text.splitlines()).strip()])
        if not prompts or not prompts[0]:
            raise ValueError(f"Prompt file is empty: {path}")
        relative = path.relative_to(root)
        for prompt_index, prompt in enumerate(prompts):
            records.append({"source": relative.as_posix(), "prompt_index": prompt_index,
                            "prompt": prompt})
    return root, records


def load_pipeline(adapter_dir, device="cuda", precision="bf16", base_model=None, local_files_only=False):
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from peft import LoraConfig
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "training_config.json").read_text(encoding="utf-8"))
    configs = json.loads((adapter_dir / "adapter_configs.json").read_text(encoding="utf-8"))
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model or cfg["pretrained_model_name_or_path"], revision=cfg["revision"], variant=cfg["variant"],
        torch_dtype=weight_dtype(precision), safety_checker=None, requires_safety_checker=False,
        local_files_only=local_files_only)
    for name, adapter_config in configs.items():
        module = getattr(pipeline, name)
        module.add_adapter(LoraConfig(**adapter_config))
        restore_adapter(module, load_file(str(adapter_dir / f"{name}_lora.safetensors")))
        module.eval()
    pipeline.tokenizer.model_max_length = cfg["max_token_length"]
    pipeline.scheduler = DDIMScheduler.from_pretrained(adapter_dir / "noise_scheduler")
    pipeline.to(device)
    return pipeline, cfg


def _prompt_limits(cfg, override):
    limit = cfg.get("max_prompt_chunks", 1) if override is None else override
    if limit < 1:
        raise ValueError("max_prompt_chunks must be positive")
    return cfg["max_token_length"], limit


@torch.no_grad()
def generate(adapter_dir, prompt, output_dir, num_images=4, seed=1337, steps=30, guidance=7.5,
             device="cuda", precision="bf16", base_model=None, local_files_only=False,
             max_prompt_chunks=None):
    pipeline, cfg = load_pipeline(adapter_dir, device, precision, base_model, local_files_only)
    chunk_length, limit = _prompt_limits(cfg, max_prompt_chunks)
    negative = [""] if guidance > 1 else None
    prompt_embeds, negative_embeds = encode_prompt_embeds(
        pipeline, [prompt], limit, chunk_length, negative)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(num_images):
        generator = torch.Generator(device=device).manual_seed(seed + index)
        image = pipeline(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                         height=cfg["resolution"], width=cfg["resolution"],
                         num_inference_steps=steps, guidance_scale=guidance, generator=generator).images[0]
        path = output / f"seed-{seed + index}.png"
        image.save(path)
        paths.append(str(path))
    atomic_json(output / "generation.json", {"adapter": str(adapter_dir), "prompt": prompt,
        "seed": seed, "steps": steps, "guidance_scale": guidance, "resolution": cfg["resolution"],
        "images": paths, "mode": "text-only", "max_prompt_chunks": limit,
        "prompt_token_count": len(prompt_token_ids(pipeline.tokenizer, prompt))})
    return paths


@torch.no_grad()
def generate_from_prompt_directory(adapter_dir, prompt_dir, output_dir, num_images_per_prompt=1,
                                   batch_size=4, seed=1337, steps=30, guidance=7.5,
                                   negative_prompt=None, device="cuda", precision="bf16",
                                   base_model=None, local_files_only=False, recursive=False,
                                   one_prompt_per_line=False, overwrite=False, skip_existing=False,
                                   max_prompt_chunks=None):
    """Load the pipeline once and generate a deterministic image set from .txt prompts."""
    if min(num_images_per_prompt, batch_size, steps) <= 0:
        raise ValueError("num_images_per_prompt, batch_size and steps must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if overwrite and skip_existing:
        raise ValueError("overwrite and skip_existing are mutually exclusive")
    prompt_root, prompt_records = read_prompt_directory(
        prompt_dir, recursive=recursive, one_prompt_per_line=one_prompt_per_line)
    pipeline, cfg = load_pipeline(adapter_dir, device, precision, base_model, local_files_only)
    chunk_length, limit = _prompt_limits(cfg, max_prompt_chunks)
    # Check every TXT before writing any image. A too-long caption must not be
    # silently shortened halfway through a large run.
    for record in prompt_records:
        source = f"{record['source']} prompt {record['prompt_index']}"
        chunks = chunk_prompt(pipeline.tokenizer, record["prompt"], chunk_length,
                              limit, source=source)
        record["prompt_token_count"] = len(prompt_token_ids(pipeline.tokenizer, record["prompt"]))
        record["prompt_chunks"] = len(chunks)
    if negative_prompt is not None and guidance > 1:
        chunk_prompt(pipeline.tokenizer, negative_prompt, chunk_length, limit,
                     source="negative_prompt")
    output = Path(output_dir).resolve()
    jobs = []
    for record_index, record in enumerate(prompt_records):
        source = Path(record["source"])
        # Preserve only the prompt directory hierarchy (for example the 17 class
        # folders). Do not create one output directory for every TXT file.
        destination = output / source.parent
        for image_index in range(num_images_per_prompt):
            image_seed = seed + record_index * num_images_per_prompt + image_index
            filename = (f"{source.stem}_prompt-{record['prompt_index']:03d}_"
                        f"image-{image_index:03d}_seed-{image_seed}.png")
            jobs.append({**record, "image_index": image_index, "seed": image_seed,
                         "image": str(destination / filename)})
    conflicts = [job["image"] for job in jobs if Path(job["image"]).exists()]
    if conflicts and not (overwrite or skip_existing):
        raise FileExistsError(
            f"{len(conflicts)} output images already exist; choose a new --output, "
            "or use --skip-existing/--overwrite. Example: {conflicts[0]}")
    pending = [job for job in jobs if overwrite or not Path(job["image"]).exists()]
    manifest = {
        "adapter": str(Path(adapter_dir).resolve()), "prompt_dir": str(prompt_root),
        "output_dir": str(output), "resolution": cfg["resolution"], "steps": steps,
        "guidance_scale": guidance, "negative_prompt": negative_prompt, "base_seed": seed,
        "precision": precision, "text_encoding": "clip-chunk-v1",
        "num_images_per_prompt": num_images_per_prompt, "batch_size": batch_size,
        "recursive": recursive, "one_prompt_per_line": one_prompt_per_line,
        "max_prompt_chunks": limit, "chunk_length": chunk_length,
        "mode": "text-only-batch", "total_prompts": len(prompt_records),
        "total_images": len(jobs), "completed_images": len(jobs) - len(pending), "images": jobs,
    }
    if conflicts and skip_existing:
        old_manifest_path = output / "generation_manifest.json"
        if not old_manifest_path.is_file():
            raise ValueError("Cannot skip existing images without their original manifest; use a new output directory")
        old = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        comparison_keys = ("adapter", "resolution", "steps", "guidance_scale",
                           "negative_prompt", "base_seed", "precision", "text_encoding",
                           "num_images_per_prompt", "max_prompt_chunks", "chunk_length", "images")
        if any(old.get(key) != manifest[key] for key in comparison_keys):
            raise ValueError("Existing images were generated with different prompts, text encoding "
                             "or settings; use a new output directory or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "generation_manifest.json", manifest)
    for start in range(0, len(pending), batch_size):
        group = pending[start:start + batch_size]
        for job in group:
            Path(job["image"]).parent.mkdir(parents=True, exist_ok=True)
        generators = [torch.Generator(device=device).manual_seed(job["seed"]) for job in group]
        negatives = ([negative_prompt or ""] * len(group) if guidance > 1 else None)
        prompt_embeds, negative_embeds = encode_prompt_embeds(
            pipeline, [job["prompt"] for job in group], limit, chunk_length,
            negatives)
        images = pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            height=cfg["resolution"], width=cfg["resolution"],
            num_inference_steps=steps, guidance_scale=guidance,
            generator=generators,
        ).images
        for job, image in zip(group, images):
            image.save(job["image"])
        manifest["completed_images"] += len(group)
        atomic_json(output / "generation_manifest.json", manifest)
    return manifest
