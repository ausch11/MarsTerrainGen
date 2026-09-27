"""Independent, versioned latent/teacher caches. No per-sample silent fallback."""
import json
import os
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import canonical_image, pixels_from_image
from .utils import atomic_json, fingerprint, model_identity, seed_for


class CacheStore:
    def __init__(self, root, specs):
        self.root, self.specs = Path(root), specs
        self.signatures = {k: fingerprint(v) for k, v in specs.items()}

    def path(self, record, kind):
        # Includes complete filename, class path and image CONTENT hash; no stem collision.
        key = fingerprint((record.relative_path, record.image_hash))
        return self.root / kind / self.signatures[kind] / key[:2] / (key + ".npz")

    def _read(self, record, kind):
        path = self.path(record, kind)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            array = archive["value"].copy()
        expected = self.specs[kind]["shape"]
        if metadata != {"signature": self.signatures[kind], "source": record.image_hash}:
            raise ValueError(f"Cache metadata mismatch: {path}")
        if list(array.shape) != list(expected) or not np.isfinite(array).all():
            raise ValueError(f"Invalid {kind} shape/value in {path}: {array.shape}, expected {expected}")
        return array

    def valid(self, record, kind):
        try:
            self._read(record, kind)
            return True
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            return False

    def load(self, record, kind):
        try:
            return torch.from_numpy(self._read(record, kind)).float()
        except Exception as exc:
            raise RuntimeError(f"Missing/corrupt {kind} cache for {record.image}; rerun cache preparation") from exc

    def write(self, record, kind, value):
        path = self.path(record, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.asarray(value, dtype=np.float32)
        if list(array.shape) != self.specs[kind]["shape"] or not np.isfinite(array).all():
            raise ValueError(f"Refusing invalid {kind} cache for {record.image}: {array.shape}")
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        with open(tmp, "wb") as stream:
            np.savez(stream, value=array, metadata=json.dumps({
                "signature": self.signatures[kind], "source": record.image_hash}))
        os.replace(tmp, path)

    def save_specs(self):
        for kind, spec in self.specs.items():
            atomic_json(self.root / kind / self.signatures[kind] / "spec.json", spec)


def make_cache(cfg, vae, teacher=None):
    spatial_scale = 2 ** (len(vae.config.block_out_channels) - 1)
    if cfg.resolution % spatial_scale:
        raise ValueError("Resolution is incompatible with VAE downsampling")
    common = {"schema": 1, "tag": cfg.cache_tag, "resolution": cfg.resolution,
              "center_crop": cfg.center_crop, "seed": cfg.seed,
              "crop": "content-seeded-single-crop", "resample": "PIL-bicubic", "exif_transpose": True}
    specs = {}
    if cfg.cache_latents:
        specs["latent"] = {**common, "vae": model_identity(
            cfg.pretrained_model_name_or_path, cfg.revision, "vae", dict(vae.config)),
            "dtype": cfg.vae_dtype, "scaling_factor": float(vae.config.scaling_factor),
            "posterior": "one-content-seeded-sample", "shape": [vae.config.latent_channels,
            cfg.resolution // spatial_scale, cfg.resolution // spatial_scale]}
    if teacher is not None:
        specs["feature"] = {**common, "teacher": teacher.identity, "shape": [teacher.hidden_size]}
    return CacheStore(cfg.cache_dir, specs)


@torch.no_grad()
def build_cache(records, cfg, cache, vae, teacher, device, kinds=None):
    kinds = list(cache.specs) if kinds is None else kinds
    cache.save_specs()
    vae.eval()
    vae_dtype = getattr(torch, cfg.vae_dtype)
    for kind in kinds:
        pending = [r for r in records if not cache.valid(r, kind)]
        print(f"[cache:{kind}] total={len(records)}, valid={len(records)-len(pending)}, rebuild={len(pending)}")
        for start in tqdm(range(0, len(pending), cfg.cache_batch_size), desc=f"cache {kind}"):
            group = pending[start:start + cfg.cache_batch_size]
            images = [canonical_image(r, cfg.resolution, cfg.center_crop, cfg.seed) for r in group]
            if kind == "feature":
                values = teacher.encode(images)
            else:
                pixels = torch.stack([pixels_from_image(im) for im in images]).to(device, dtype=vae_dtype)
                posterior = vae.encode(pixels).latent_dist
                # Per-image independent RNG makes cold/warm/partial cache runs equivalent.
                eps = torch.stack([torch.randn(posterior.mean.shape[1:], generator=torch.Generator().manual_seed(
                    seed_for(cfg.seed, r.image_hash, "posterior"))) for r in group]).to(device, dtype=vae_dtype)
                values = (posterior.mean + posterior.std * eps) * vae.config.scaling_factor
                values = values.float().cpu()
            for record, value in zip(group, values):
                cache.write(record, kind, value.numpy())
