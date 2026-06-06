from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config  
from src.data import build_text_dataset, load_jsonl
from src.model import add_lora, load_model_and_tokenizer
from src.train import build_trainer, save_adapter, save_log_history, train_resumable
from src.visualize import plot_loss_curves, plot_lr_schedule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)

    model, tokenizer = load_model_and_tokenizer(cfg)
    model = add_lora(cfg, model)

    train_recs = load_jsonl(cfg.file_in("data_dir", "train.jsonl"))
    val_recs = load_jsonl(cfg.file_in("data_dir", "val.jsonl"))
    train_ds = build_text_dataset(train_recs, tokenizer)
    eval_ds = build_text_dataset(val_recs, tokenizer)

    trainer = build_trainer(cfg, model, tokenizer, train_ds, eval_ds)
    train_resumable(cfg, trainer)

    log_path = save_log_history(cfg, trainer)
    save_adapter(cfg, model, tokenizer)

    plot_loss_curves(log_path, cfg.file_in("figures_dir", "loss_curves.png"))
    plot_lr_schedule(log_path, cfg.file_in("figures_dir", "lr_schedule.png"))


if __name__ == "__main__":
    main()
