from dataclasses import replace
import json

import pytest
from safetensors.torch import load_file
import torch

from mars_lora.inference import generate, load_pipeline
from mars_lora.teacher import DinoTeacher
from mars_lora.training import train


def test_tiny_b1_train_resume_export_and_text_only_generation(tiny_workspace):
    cfg, root = tiny_workspace
    teacher = DinoTeacher(cfg, "cpu")
    assert not any(p.requires_grad for p in teacher.model.parameters())
    del teacher
    uninterrupted = train(cfg)
    assert uninterrupted["global_step"] == 4
    saved_state = json.loads((root / "run/checkpoint-4/trainer_state.json").read_text(encoding="utf-8"))
    assert saved_state["best_validation_loss"] == uninterrupted["best_validation_loss"]
    assert (root / "run/final/unet_lora.safetensors").exists()
    assert not (root / "run/final/projector.safetensors").exists()
    assert (root / "run/checkpoint-4/projector.safetensors").exists()
    assert len(list((root / "run/validation/step-4").glob("*.png"))) == 2
    records = [json.loads(line) for line in (root / "run/metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(r.get("train/alignment_loss", 0) > 0 for r in records)
    assert any("validation/denoising_loss" in r for r in records)
    resumed_cfg = replace(cfg, output_dir=str(root / "resumed"))
    partial = train(resumed_cfg, stop_after_steps=1)
    assert partial["interrupted"] and partial["next_batch"] == 2
    finished = train(replace(resumed_cfg, resume_from_checkpoint="latest"))
    assert finished["global_step"] == 4 and not finished["interrupted"]
    for filename in ("unet_lora.safetensors", "text_encoder_lora.safetensors"):
        left = load_file(str(root / "run/final" / filename))
        right = load_file(str(root / "resumed/final" / filename))
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=1e-5, atol=1e-6)
    lp = load_file(str(root / "run/checkpoint-4/projector.safetensors"))
    rp = load_file(str(root / "resumed/checkpoint-4/projector.safetensors"))
    for key in lp:
        torch.testing.assert_close(lp[key], rp[key], rtol=1e-5, atol=1e-6)
    # Removing access to the teacher proves the inference bundle has no target-image dependency.
    teacher_path = root / "teacher"
    offline_teacher = root / "teacher-unavailable"
    teacher_path.rename(offline_teacher)
    paths = generate(root / "run/final", "Mars ridges", root / "generated", num_images=1,
                     steps=2, device="cpu", precision="no", local_files_only=True)
    from PIL import Image
    assert Image.open(paths[0]).size == (32, 32)


def test_tiny_baseline_never_loads_teacher(tiny_workspace):
    cfg, root = tiny_workspace
    cfg = replace(cfg, b1_enabled=False, teacher_model_name_or_path="missing-teacher-must-not-load",
                  output_dir=str(root / "baseline"), num_validation_images=0, max_train_steps=1)
    result = train(cfg)
    assert result["global_step"] == 1
    assert not (root / "baseline/checkpoint-1/projector.safetensors").exists()
    state = json.loads((root / "baseline/run.json").read_text(encoding="utf-8"))
    assert state["trainable"]["projector"]["parameters"] == 0


def test_real_dataloader_workers_do_not_contain_gpu_models(tiny_workspace):
    cfg, root = tiny_workspace
    cfg = replace(cfg, output_dir=str(root / "workers"), dataloader_num_workers=2,
                  persistent_data_loader_workers=True, num_validation_images=0, max_train_steps=3)
    result = train(cfg)
    assert result["global_step"] == 3
