from __future__ import annotations

import argparse
import gc
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data import load_jsonl
from src.evaluate import make_hf_generate_fn, run_eval
from src.visualize import plot_error_breakdown, plot_metric_comparison


def _free(model) -> None:
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def _eval_one(cfg, model_name_or_dir, label, test_recs):
    from src.model import for_inference, load_model_and_tokenizer
    cfg.model["base_id"] = model_name_or_dir
    model, tokenizer = load_model_and_tokenizer(cfg)
    for_inference(model)
    gen = make_hf_generate_fn(cfg, model, tokenizer)
    metrics = run_eval(cfg, test_recs, tokenizer, gen, label)
    _free(model)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    test_recs = load_jsonl(cfg.file_in("data_dir", "test.jsonl"))

    base_id = cfg.model["base_id"]
    adapter_dir = str(cfg.path("adapter_dir"))

    results = {
        "base": _eval_one(cfg, base_id, "base", test_recs),
        "finetuned": _eval_one(cfg, adapter_dir, "finetuned", test_recs),
    }

    plot_metric_comparison(results, cfg.file_in("figures_dir", "metric_comparison.png"))
    plot_error_breakdown(results, cfg.file_in("figures_dir", "error_breakdown.png"))


if __name__ == "__main__":
    main()
