"""Notebook cell 3: explicit sample discovery, deterministic image/caption processing."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, Sampler

from .utils import file_sha256, fingerprint, seed_for
from .text import chunk_prompt

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    image: Path
    caption: Path
    relative_path: str
    class_name: str
    image_hash: str
    caption_hash: str


def discover_images(root, caption_extension=".txt", strict=True, exclude_dirs=()):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    excludes = [Path(p).resolve() for p in exclude_dirs if p]
    records, missing = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if any(p.resolve().is_relative_to(e) for e in excludes):
            continue
        cap = p.with_suffix(caption_extension)
        if not cap.is_file() or not cap.read_text(encoding="utf-8").strip():
            missing.append(str(p))
            continue
        rel = p.relative_to(root)
        label = rel.parts[0] if len(rel.parts) > 1 else "__root__"
        records.append(ImageRecord(p, cap, rel.as_posix(), label,
                                   file_sha256(p), file_sha256(cap)))
    if strict and missing:
        raise ValueError(f"{len(missing)} images have missing/empty captions. Examples: {missing[:5]}")
    if not records:
        raise ValueError(f"No captioned images in {root}")
    return records, missing


def data_summary(records, repeats=None):
    repeats = repeats or {}
    counts = Counter(r.class_name for r in records)
    unknown = set(repeats) - set(counts)
    if unknown:
        raise ValueError(f"Repeat configuration names unknown classes: {unknown}")
    return {"unique_images": len(records), "num_classes": len(counts),
            "per_class": dict(counts),
            "sampled_per_class": {k: v * repeats.get(k, 1) for k, v in counts.items()},
            "samples_with_repeats": sum(v * repeats.get(k, 1) for k, v in counts.items())}


def validate_counts(summary, cfg):
    for expected, actual in ((cfg.expected_unique_images, summary["unique_images"]),
                             (cfg.expected_num_classes, summary["num_classes"])):
        if expected is not None and actual != expected:
            raise ValueError(f"Dataset count mismatch: expected {expected}, got {actual}. Summary: {summary}")
    if cfg.expected_images_per_class is not None:
        bad = {k: n for k, n in summary["per_class"].items() if n != cfg.expected_images_per_class}
        if bad:
            raise ValueError(f"Unexpected per-class image counts: {bad}")


def records_fingerprint(records):
    return fingerprint([(r.relative_path, r.class_name, r.image_hash, r.caption_hash) for r in records])


def canonical_image(record, resolution, center_crop, seed):
    """One reproducible crop per image; teacher and VAE see exactly the same field of view."""
    with Image.open(record.image) as src:
        image = ImageOps.exif_transpose(src).convert("RGB")
    w, h = image.size
    if w <= h:
        new_size = (resolution, max(resolution, round(h * resolution / w)))
    else:
        new_size = (max(resolution, round(w * resolution / h)), resolution)
    image = image.resize(new_size, Image.Resampling.BICUBIC)
    dw, dh = new_size[0] - resolution, new_size[1] - resolution
    if center_crop:
        x, y = dw // 2, dh // 2
    else:
        rng = random.Random(seed_for(seed, record.image_hash, "crop"))
        x, y = rng.randint(0, dw), rng.randint(0, dh)
    return image.crop((x, y, x + resolution, y + resolution))


def pixels_from_image(image):
    array = np.array(image, dtype=np.float32, copy=True) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def process_caption(caption, shuffle, keep_tokens, rng, dropout=0.0):
    if rng.random() < dropout:
        return ""
    if not shuffle:
        return caption.strip()
    tokens = [s.strip() for s in caption.split(",") if s.strip()]
    fixed, remaining = tokens[:keep_tokens], tokens[keep_tokens:]
    rng.shuffle(remaining)
    return ", ".join(fixed + remaining)


class EpochSampler(Sampler):
    """Carries the epoch inside indices, including with persistent/spawn workers."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter((self.epoch, i) for i in torch.randperm(len(self), generator=generator).tolist())


class CaptionedImageDataset(Dataset):
    def __init__(self, records, tokenizer, cfg, cache, training=True):
        self.records, self.tokenizer, self.cfg = records, tokenizer, cfg
        self.cache, self.training = cache, training
        repeats = cfg.repeats if training else {}
        self.indices = [i for i, r in enumerate(records) for _ in range(repeats.get(r.class_name, 1))]
        # CacheStore contains only paths/specs, never a CUDA model.

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        epoch, slot = index if isinstance(index, tuple) else (0, index)
        record = self.records[self.indices[slot]]
        caption = record.caption.read_text(encoding="utf-8").strip()
        caption = process_caption(caption, self.training and self.cfg.shuffle_caption,
                                  self.cfg.keep_tokens,
                                  random.Random(seed_for(self.cfg.seed, epoch, slot, "caption")),
                                  self.cfg.caption_dropout if self.training else 0.0)
        ids = chunk_prompt(self.tokenizer, caption, self.cfg.max_token_length,
                           self.cfg.max_prompt_chunks, source=str(record.caption))
        latents = self.cache.load(record, "latent") if self.cfg.cache_latents else None
        features = self.cache.load(record, "feature") if self.cfg.b1_enabled and self.training else None
        pixels = None if self.cfg.cache_latents else pixels_from_image(canonical_image(
            record, self.cfg.resolution, self.cfg.center_crop, self.cfg.seed))
        return {"input_ids": ids, "latents": latents, "pixel_values": pixels,
                "teacher_features": features, "image_path": str(record.image), "caption": caption}


def collate_fn(examples):
    result = {}
    # The longest caption in the batch determines its cross-attention sequence.
    # Empty CLIP chunks make the shape uniform without throwing away any tokens.
    ids = [e["input_ids"] for e in examples]
    length = ids[0].shape[1]
    bos, eos = ids[0][0, 0].item(), ids[0][0, -1].item()
    empty = ids[0].new_tensor([bos] + [eos] * (length - 1))
    max_chunks = max(item.shape[0] for item in ids)
    result["input_ids"] = torch.stack([
        torch.cat([item, empty.expand(max_chunks - item.shape[0], -1)], dim=0)
        for item in ids])
    for key in ("latents", "pixel_values", "teacher_features"):
        values = [e[key] for e in examples]
        if all(v is None for v in values):
            result[key] = None
        elif any(v is None for v in values):
            raise ValueError(f"Mixed missing/present {key}: cache completeness is required")
        else:
            result[key] = torch.stack(values)
    result["image_paths"] = [e["image_path"] for e in examples]
    result["captions"] = [e["caption"] for e in examples]
    return result
