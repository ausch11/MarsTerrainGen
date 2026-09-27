from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from mars_lora.cache import CacheStore
from mars_lora.config import TrainConfig, load_config
from mars_lora.data import (EpochSampler, canonical_image, collate_fn, data_summary,
                           discover_images, process_caption)
from conftest import make_pair


def test_flat_and_numeric_directories_not_repeated(tmp_path):
    make_pair(tmp_path, "root", 1)
    make_pair(tmp_path / "0_crater", "a", 2)
    make_pair(tmp_path / "17_ridges", "b", 3)
    (tmp_path / "latents_cache").mkdir()
    records, _ = discover_images(tmp_path)
    summary = data_summary(records)
    assert summary["unique_images"] == summary["samples_with_repeats"] == 3
    assert summary["sampled_per_class"] == {"__root__": 1, "0_crater": 1, "17_ridges": 1}


def test_missing_caption_is_not_silent(tmp_path):
    Image.new("RGB", (32, 32)).save(tmp_path / "missing.png")
    with pytest.raises(ValueError, match="missing/empty captions"):
        discover_images(tmp_path)


def test_cache_versions_content_and_extension(tmp_path):
    make_pair(tmp_path / "data", "a", 1)
    records, _ = discover_images(tmp_path / "data")
    record = records[0]
    cache = CacheStore(tmp_path / "cache", {"feature": {"shape": [4], "version": 1}})
    cache.write(record, "feature", np.ones(4))
    assert cache.valid(record, "feature")
    assert torch.equal(cache.load(record, "feature"), torch.ones(4))
    other_ext = replace(record, relative_path="a.jpg")
    assert cache.path(record, "feature") != cache.path(other_ext, "feature")
    changed = replace(record, image_hash="new-content")
    assert not cache.valid(changed, "feature")
    new_version = CacheStore(tmp_path / "cache", {"feature": {"shape": [4], "version": 2}})
    assert not new_version.valid(record, "feature")
    cache.path(record, "feature").write_bytes(b"interrupted file")
    assert not cache.valid(record, "feature")
    with pytest.raises(RuntimeError, match="Missing/corrupt"):
        cache.load(record, "feature")


def test_mixed_cache_batch_is_explicit_error():
    common = {"input_ids": torch.ones(3), "teacher_features": None, "image_path": "a", "caption": "a"}
    examples = [{**common, "latents": torch.ones(4, 2, 2), "pixel_values": None},
                {**common, "latents": None, "pixel_values": torch.ones(3, 16, 16)}]
    with pytest.raises(ValueError, match="Mixed missing/present latents"):
        collate_fn(examples)


def test_crop_is_deterministic_and_shared(tmp_path):
    make_pair(tmp_path, "a", 3)
    records, _ = discover_images(tmp_path)
    image1 = canonical_image(records[0], 16, False, 1337)
    torch.rand(100)
    image2 = canonical_image(records[0], 16, False, 1337)
    assert np.array_equal(image1, image2)


def test_sampler_resume_and_epoch_identity():
    sampler = EpochSampler(list(range(20)), 1337)
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    assert first == list(sampler)
    assert all(epoch == 3 for epoch, index in first)
    sampler.set_epoch(4)
    assert [i for _, i in first] != [i for _, i in sampler]


def test_config_rejects_unknown_and_incompatible_augmentation(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"not_an_option": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
    with pytest.raises(ValueError, match="random_flip"):
        TrainConfig(random_flip=True).validate()

