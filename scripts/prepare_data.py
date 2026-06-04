"""CLI: download, normalize, split, and cache the dataset as JSONL.

Usage:
    python scripts/prepare_data.py [--config config/config.yaml]

Writes train.jsonl / val.jsonl / test.jsonl into the configured data dir.
Needs no GPU (but may need `huggingface-cli login` for the gated dataset).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data import load_records, save_jsonl, split_records  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    records = load_records(cfg)
    splits = split_records(cfg, records)

    for name, recs in splits.items():
        out = cfg.file_in("data_dir", f"{name}.jsonl")
        save_jsonl(recs, out)
        print(f"[prepare] wrote {len(recs):>6d} rows -> {out}")


if __name__ == "__main__":
    main()
