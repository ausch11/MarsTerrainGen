"""Build and verify a source-only delivery archive, without models or test caches."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root.parent / "mars_lora_b1.zip"
    output = output.resolve()
    allowed_dirs = {"mars_lora", "configs", "tests", "tools"}
    allowed_suffixes = {".py", ".json", ".md", ".txt", ".toml", ".xml"}
    contents = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or any(part.startswith(".") or part == "__pycache__" for part in rel.parts[:-1]):
            continue
        if len(rel.parts) > 1 and rel.parts[0] not in allowed_dirs:
            continue
        if path.suffix not in allowed_suffixes and path.name != ".gitignore":
            continue
        data = path.read_bytes()
        if path.suffix == ".py":
            ast.parse(data.decode("utf-8"), filename=str(rel))
        contents[f"{root.name}/{rel.as_posix()}"] = data
    required = {"README.md", "TEST_REPORT.md", "test-results.xml", "train.py", "generate.py", "configs/b1.json"}
    if not all(f"{root.name}/{name}" in contents for name in required):
        raise RuntimeError("Delivery is missing required files")
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    contents[f"{root.name}/PACKAGE_MANIFEST.json"] = json.dumps(manifest, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(contents):
            raise RuntimeError("Archive integrity failed")
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise RuntimeError(f"Archive content mismatch: {name}")
    print(json.dumps({"archive": str(output), "files": len(contents), "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "verification": "all Python ASTs parsed; ZIP CRC and per-file SHA256 verified"}, indent=2))


if __name__ == "__main__":
    main()
