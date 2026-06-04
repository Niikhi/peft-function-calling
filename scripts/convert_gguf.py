from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workdir", default=None, help="where to clone/build llama.cpp")
    args = ap.parse_args()

    cfg = load_config(args.config)
    workdir = args.workdir or str(cfg.root)

    # Load base + trained adapter through Unsloth, then merge to fp16.
    from src.gguf_convert import merge_adapter_unsloth, run_full_conversion
    from src.model import load_model_and_tokenizer

    cfg.model["base_id"] = str(cfg.path("adapter_dir"))  # loads base + adapter
    model, tokenizer = load_model_and_tokenizer(cfg)
    merged_dir = merge_adapter_unsloth(cfg, model, tokenizer)

    rows = run_full_conversion(cfg, tokenizer, merged_dir, workdir)

    manifest = cfg.file_in("metrics_dir", "gguf_manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[convert] wrote manifest -> {manifest}")


if __name__ == "__main__":
    main()
