import argparse
import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .config import load_config


def _overrides(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Overrides use --set key=value")
        key, value = item.split("=", 1)
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    return result


def train_main():
    parser = argparse.ArgumentParser(description="SD1.5 LoRA baseline / B1 target-feature supervision")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--stop-after-steps", type=int,
                        help="Gracefully checkpoint and stop at this global step, keeping the full LR schedule")
    args = parser.parse_args()
    from .training import train
    result = train(load_config(args.config, _overrides(args.set)), args.cache_only, args.stop_after_steps)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def inspect_main():
    parser = argparse.ArgumentParser(description="Audit actual sample counts, labels and caption matching")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    from .training import inspect_data
    _, val, summary = inspect_data(load_config(args.config, _overrides(args.set)))
    print(json.dumps({**summary, "validation_images": len(val)}, ensure_ascii=False, indent=2))


def generate_main():
    parser = argparse.ArgumentParser(description="Text-only generation; no teacher or target-image input")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/generated")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["no", "bf16", "fp16"], default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    from .inference import generate
    paths = generate(args.adapter, args.prompt, args.output, args.num_images, args.seed,
        args.steps, args.guidance_scale, args.device, args.precision, args.base_model, args.local_files_only)
    print(json.dumps(paths, ensure_ascii=False, indent=2))


def batch_generate_main():
    parser = argparse.ArgumentParser(
        description="Batch text-only generation from a directory of UTF-8 .txt prompt files")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompts-dir", required=True)
    parser.add_argument("--output", default="outputs/generated_batch")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--num-images-per-prompt", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Images generated simultaneously; reduce this if CUDA runs out of memory")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["no", "bf16", "fp16"], default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--recursive", action="store_true",
                        help="Read .txt files in all subdirectories")
    parser.add_argument("--one-prompt-per-line", action="store_true",
                        help="Treat each non-empty line as an independent prompt")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--skip-existing", action="store_true")
    outputs.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from .inference import generate_from_prompt_directory
    result = generate_from_prompt_directory(
        adapter_dir=args.adapter, prompt_dir=args.prompts_dir, output_dir=args.output,
        num_images_per_prompt=args.num_images_per_prompt, batch_size=args.batch_size,
        seed=args.seed, steps=args.steps, guidance=args.guidance_scale,
        negative_prompt=args.negative_prompt, device=args.device, precision=args.precision,
        base_model=args.base_model, local_files_only=args.local_files_only,
        recursive=args.recursive, one_prompt_per_line=args.one_prompt_per_line,
        overwrite=args.overwrite, skip_existing=args.skip_existing)
    print(json.dumps({k: v for k, v in result.items() if k != "images"},
                     ensure_ascii=False, indent=2))
