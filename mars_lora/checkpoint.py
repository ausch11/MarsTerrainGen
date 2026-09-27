"""Notebook cells 4/8/9: LoRA + projector export and resumable optimizer/RNG state."""
import json
import re
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

from .utils import atomic_json, fingerprint, jsonable


def cpu_state(state):
    return {k: v.detach().float().contiguous().cpu() for k, v in state.items()}


def adapter_state(model):
    from peft.utils import get_peft_model_state_dict
    # Token embeddings are frozen and never resized in this project. Avoid PEFT's
    # automatic vocabulary probe against the root of a composite Diffusers model.
    return get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)


def restore_adapter(model, state):
    from peft.utils import set_peft_model_state_dict
    expected = adapter_state(model)
    if set(expected) != set(state):
        raise ValueError(f"LoRA keys mismatch; missing={set(expected)-set(state)}, unexpected={set(state)-set(expected)}")
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"LoRA shape mismatch: {key}")
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    if result and getattr(result, "unexpected_keys", None):
        raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")


def save_bundle(model, cfg, scheduler, folder, include_projector=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    configs = {"unet": jsonable(model.unet.peft_config["default"].to_dict())}
    save_file(cpu_state(adapter_state(model.unet)), str(folder / "unet_lora.safetensors"))
    if cfg.train_text_encoder_lora:
        configs["text_encoder"] = jsonable(model.text_encoder.peft_config["default"].to_dict())
        save_file(cpu_state(adapter_state(model.text_encoder)), str(folder / "text_encoder_lora.safetensors"))
    atomic_json(folder / "adapter_configs.json", configs)
    atomic_json(folder / "training_config.json", cfg.to_dict())
    # Keep the training prediction_type at inference, including v_prediction.
    scheduler.save_pretrained(folder / "noise_scheduler")
    if include_projector and model.projector is not None:
        save_file(cpu_state(model.projector.state_dict()), str(folder / "projector.safetensors"))


def load_bundle_weights(model, folder, include_projector=False):
    folder = Path(folder)
    configs = json.loads((folder / "adapter_configs.json").read_text(encoding="utf-8"))
    for name in ("unet", "text_encoder"):
        module = getattr(model, name)
        current = getattr(module, "peft_config", {}).get("default")
        if (current is None) != (name not in configs):
            raise ValueError(f"Adapter enablement mismatch for {name}")
        if current is None:
            continue
        for key in ("r", "lora_alpha", "lora_dropout", "use_rslora", "use_dora", "bias", "target_modules"):
            actual, saved = jsonable(current.to_dict().get(key)), configs[name].get(key)
            if key == "target_modules":
                actual, saved = sorted(actual), sorted(saved)
            if actual != saved:
                raise ValueError(f"Adapter configuration mismatch: {name}.{key}")
        restore_adapter(module, load_file(str(folder / f"{name}_lora.safetensors")))
    if include_projector and model.projector is not None:
        model.projector.load_state_dict(load_file(str(folder / "projector.safetensors")), strict=True)


def register_hooks(accelerator, model, cfg, noise_scheduler):
    def save_hook(models, weights, output_dir):
        weights.clear()  # Do not serialize frozen SD backbones.
        save_bundle(accelerator.unwrap_model(model), cfg, noise_scheduler, output_dir, include_projector=True)

    def load_hook(models, input_dir):
        from .models import cast_trainable_fp32
        raw = accelerator.unwrap_model(model)
        load_bundle_weights(raw, input_dir, include_projector=True)
        cast_trainable_fp32(raw)
        models.clear()

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)


def run_signature(cfg, data_id, cache_specs):
    recipe = cfg.to_dict()
    # Old checkpoints predate this field and used exactly one CLIP window.
    if recipe.get("max_prompt_chunks") == 1:
        recipe.pop("max_prompt_chunks")
    # Reporting, workers and checkpoint frequency may change without changing optimization.
    for key in ("output_dir", "cache_dir", "resume_from_checkpoint", "report_to", "log_every",
                "dataloader_num_workers", "persistent_data_loader_workers", "checkpointing_epochs",
                "checkpointing_steps", "checkpoints_total_limit", "validation_epochs",
                "num_validation_images", "local_files_only"):
        recipe.pop(key, None)
    return fingerprint({"recipe": recipe, "data": data_id, "cache": cache_specs})


def resolve_resume(output_dir, requested):
    if not requested:
        return None
    output_dir = Path(output_dir)
    if requested != "latest":
        candidate = Path(requested)
        if not candidate.is_dir():
            candidate = output_dir / requested
        if not (candidate / "COMPLETE").is_file():
            raise FileNotFoundError(f"Incomplete/missing checkpoint: {candidate}")
        return candidate
    candidates = checkpoint_dirs(output_dir)
    if not candidates:
        raise FileNotFoundError("resume=latest requested but no complete checkpoint exists")
    return candidates[-1]


def checkpoint_dirs(output_dir):
    root = Path(output_dir)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir()
                   and re.fullmatch(r"checkpoint-\d+", p.name) and (p / "COMPLETE").is_file()),
                  key=lambda p: int(p.name.split("-")[-1]))


def save_checkpoint(accelerator, cfg, state):
    root = Path(cfg.output_dir).resolve()
    dest = root / f"checkpoint-{state['global_step']}"
    if (dest / "COMPLETE").exists():
        # Validation can update best_validation_loss after this same optimizer step.
        atomic_json(dest / "trainer_state.json", state)
        return dest
    accelerator.save_state(str(dest))
    atomic_json(dest / "trainer_state.json", state)
    (dest / "COMPLETE").write_text("complete\n", encoding="utf-8")
    paths = checkpoint_dirs(root)
    for old in paths[:-cfg.checkpoints_total_limit]:
        resolved = old.resolve()
        if resolved.parent != root or not re.fullmatch(r"checkpoint-\d+", resolved.name):
            raise RuntimeError(f"Refusing unsafe checkpoint cleanup: {resolved}")
        shutil.rmtree(resolved)
    return dest
