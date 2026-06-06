from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def is_kaggle() -> bool:
    return os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def is_colab() -> bool:
    return "COLAB_GPU" in os.environ or os.path.exists("/content")


def writable_root() -> Path:
    if is_kaggle():
        return Path("/kaggle/working")
    return PROJECT_ROOT


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path = field(default_factory=writable_root)

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def lora(self) -> dict[str, Any]:
        return self.raw["lora"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def hub(self) -> dict[str, Any]:
        """Checkpoint-backup settings. repo_id falls back to env HF_HUB_REPO_ID,
        and push is auto-disabled if no repo is resolved."""
        h = dict(self.raw.get("hub") or {})
        h["repo_id"] = h.get("repo_id") or os.environ.get("HF_HUB_REPO_ID")
        h["push_to_hub"] = bool(h.get("push_to_hub")) and bool(h["repo_id"])
        return h

    @property
    def gguf(self) -> dict[str, Any]:
        return self.raw["gguf"]

    def path(self, key: str) -> Path:
        rel = self.raw["paths"][key]
        p = Path(rel)
        if not p.is_absolute():
            p = self.root / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def file_in(self, key: str, filename: str) -> Path:
        return self.path(key) / filename


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)


if __name__ == "__main__":
    cfg = load_config()
    env = "kaggle" if is_kaggle() else "colab" if is_colab() else "local"
    print(f"Environment    : {env}")
    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Writable root  : {cfg.root}")
    print(f"Model base id  : {cfg.model['base_id']}")
    print(f"Dataset id     : {cfg.data['dataset_id']}")
    print(f"Adapter dir    : {cfg.path('adapter_dir')}")
    print(f"GGUF dir       : {cfg.path('gguf_dir')}")
