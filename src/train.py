from __future__ import annotations

import json

from .config import Config


def build_trainer(cfg: Config, model, tokenizer, train_ds, eval_ds):
    from trl import SFTConfig, SFTTrainer

    t = cfg.training
    eval_strategy = t.get("eval_strategy", "steps")

    # In-training eval is expensive. Disable it entirely ("no") for max speed, or
    # cap it to a small slice so the loss curve stays cheap. Real accuracy metrics
    # are computed afterward on the full test split.
    if eval_strategy == "no":
        eval_ds = None
    elif eval_ds is not None:
        cap = cfg.evaluation.get("eval_during_training_samples")
        if cap and len(eval_ds) > cap:
            eval_ds = eval_ds.select(range(cap))

    hub = cfg.hub
    args = SFTConfig(
        output_dir=str(cfg.path("output_dir")),
        push_to_hub=hub["push_to_hub"],
        hub_model_id=hub["repo_id"],
        # "all_checkpoints" mirrors every checkpoint-N folder to the repo, so a
        # fresh run can pull the latest one and resume.
        hub_strategy="all_checkpoints" if hub["push_to_hub"] else "every_save",
        hub_private_repo=bool(hub.get("private", True)),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        optim=t["optim"],
        logging_steps=t["logging_steps"],
        eval_strategy=eval_strategy,
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        max_grad_norm=t["max_grad_norm"],
        seed=t["seed"],
        packing=t["packing"],
        max_seq_length=cfg.model["max_seq_length"],
        dataset_text_field="text",
        report_to="none",
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
    )

    # Completion-only loss masking via Unsloth's helper (version-stable across
    # the TRL releases that removed DataCollatorForCompletionOnlyLM). If it's
    # unavailable for any reason, we fall back to plain full-text SFT — training
    # still works, the model just also sees loss on the prompt tokens.
    try:
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        print("[train] completion-only masking enabled (assistant turns only).")
    except Exception as e:  # noqa: BLE001
        print(f"[train] completion-only masking unavailable ({e}); using full-text SFT.")

    return trainer


def _restore_from_hub(cfg: Config, out_dir: str):
    """Download the latest checkpoint-N folder from the Hub repo into out_dir.

    Returns the local checkpoint path, or None if the repo has no checkpoints
    yet (or can't be reached -- in which case we just train from scratch).
    """
    repo_id = cfg.hub["repo_id"]
    try:
        from huggingface_hub import HfApi, snapshot_download

        files = HfApi().list_repo_files(repo_id)
        steps = {
            int(f.split("/")[0].split("-")[1])
            for f in files
            if f.startswith("checkpoint-") and "/" in f
        }
        if not steps:
            return None
        latest = max(steps)
        print(f"[train] Restoring checkpoint-{latest} from Hub repo '{repo_id}'.")
        snapshot_download(
            repo_id=repo_id,
            local_dir=out_dir,
            allow_patterns=[f"checkpoint-{latest}/*"],
        )
        return f"{out_dir}/checkpoint-{latest}"
    except Exception as e:  # noqa: BLE001
        print(f"[train] Hub restore skipped ({type(e).__name__}: {e}).")
        return None


def train_resumable(cfg: Config, trainer):
    """Run training, resuming from the latest checkpoint if one exists.

    Kaggle/Colab sessions die on idle or the 12h cap. Because the trainer saves
    a checkpoint every ``save_steps``, a re-run can pick up where it left off
    instead of starting over -- as long as ``output_dir`` survived (it does in
    Commit/"Save & Run All" mode, or if you mounted prior output as a dataset).
    """
    from transformers.trainer_utils import get_last_checkpoint

    out_dir = str(cfg.path("output_dir"))

    def _local_ckpt():
        try:
            return get_last_checkpoint(out_dir)
        except Exception:  # noqa: BLE001 -- no dir yet, fresh run
            return None

    last = _local_ckpt()

    # Nothing on disk (e.g. Kaggle wiped /kaggle/working) but a Hub backup exists
    # -> pull the latest checkpoint down so we can resume instead of restarting.
    if not last and cfg.hub["push_to_hub"]:
        last = _restore_from_hub(cfg, out_dir) or _local_ckpt()

    if last:
        print(f"[train] Resuming from checkpoint: {last}")
        return trainer.train(resume_from_checkpoint=last)
    print("[train] No checkpoint found -> training from scratch.")
    return trainer.train()


def save_log_history(cfg: Config, trainer) -> str:
    out = cfg.file_in("metrics_dir", "train_log_history.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"[train] Wrote log history -> {out}")
    return str(out)


def save_adapter(cfg: Config, model, tokenizer) -> str:
    adapter_dir = cfg.path("adapter_dir")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[train] Saved adapter -> {adapter_dir}")
    return str(adapter_dir)
