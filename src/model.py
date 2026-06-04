from __future__ import annotations

from .config import Config


def load_model_and_tokenizer(cfg: Config):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model["base_id"],
        max_seq_length=cfg.model["max_seq_length"],
        load_in_4bit=cfg.model["load_in_4bit"],
        dtype=cfg.model["dtype"],
    )
    return model, tokenizer


def add_lora(cfg: Config, model):
    from unsloth import FastLanguageModel

    lora = cfg.lora
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        target_modules=lora["target_modules"],
        use_gradient_checkpointing=lora["use_gradient_checkpointing"],
        random_state=cfg.training["seed"],
    )
    return model


def load_tokenizer_only(cfg: Config):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg.model["hf_id"], trust_remote_code=True)


def for_inference(model):
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    return model
