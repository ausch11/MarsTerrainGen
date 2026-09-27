import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def fingerprint(data):
    return hashlib.sha256(json.dumps(jsonable(data), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_for(*parts):
    return int(fingerprint(parts)[:15], 16)


def model_identity(identifier, revision=None, subfolder=None, model_config=None):
    root = Path(identifier)
    result = {"id": str(identifier), "revision": revision, "subfolder": subfolder,
              "config": model_config}
    if root.is_dir():
        root = root / subfolder if subfolder else root
        result["local_files"] = [(str(p.relative_to(root)), p.stat().st_size,
                                   p.stat().st_mtime_ns,
                                   file_sha256(p) if p.suffix == ".json" else None)
                                  for p in sorted(root.rglob("*"))
                                  if p.is_file() and p.suffix in {".json", ".safetensors", ".bin"}]
    else:
        # from_pretrained has already populated this cache. Pin a moving Hub revision
        # to the resolved snapshot without requiring another network request.
        from huggingface_hub import try_to_load_from_cache
        filename = f"{subfolder}/config.json" if subfolder else "config.json"
        cached = try_to_load_from_cache(str(identifier), filename, revision=revision or "main")
        if isinstance(cached, str):
            parts = Path(cached).parts
            if "snapshots" in parts:
                result["resolved_commit"] = parts[parts.index("snapshots") + 1]
    return result


def environment_info():
    result = {}
    for name in ("torch", "diffusers", "transformers", "peft", "accelerate", "numpy", "Pillow"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


@contextlib.contextmanager
def isolated_rng():
    import numpy as np
    import torch
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
