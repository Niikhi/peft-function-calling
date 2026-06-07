from __future__ import annotations
import subprocess
from pathlib import Path

from .config import Config
from .data import load_jsonl, to_openai_tool
from .prompts import build_prompt_text


def _run(cmd: list[str], **kw) -> None:
    print("[gguf] $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)


def merge_adapter_unsloth(cfg: Config, model, tokenizer) -> str:
    merged = cfg.path("merged_dir")
    model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
    print(f"[gguf] merged fp16 model -> {merged}")
    return str(merged)

def build_llama_cpp(cfg: Config, workdir) -> Path:
    workdir = Path(workdir)
    repo = workdir / "llama.cpp"
    if not repo.exists():
        _run(["git", "clone", "--depth", "1", cfg.gguf["llama_cpp_repo"], str(repo)])
    _run(["cmake", "-B", str(repo / "build"), "-S", str(repo), "-DGGML_CUDA=OFF"])
    _run(["cmake", "--build", str(repo / "build"), "--config", "Release", "-j"])
    return repo


def _bin(repo: Path, name: str) -> str:
    """Locate a built llama.cpp binary regardless of build layout."""
    for cand in [repo / "build" / "bin" / name, repo / "build" / name, repo / name]:
        if cand.exists():
            return str(cand)
    return str(repo / "build" / "bin" / name) 

def write_calibration_file(cfg: Config, tokenizer) -> str:
    train = load_jsonl(cfg.file_in("data_dir", "train.jsonl"))
    n = cfg.gguf["imatrix_samples"]
    out = cfg.file_in("gguf_dir", "calibration.txt")
    with open(out, "w", encoding="utf-8") as f:
        for rec in train[:n]:
            tools = [to_openai_tool(t) for t in rec["tools"]]
            f.write(build_prompt_text(tokenizer, rec["query"], tools) + "\n")
    print(f"[gguf] wrote calibration corpus ({min(n, len(train))} rows) -> {out}")
    return str(out)


def convert_to_f16(repo: Path, merged_dir: str, gguf_dir: Path) -> str:
    out = gguf_dir / "model-f16.gguf"
    _run(["python", str(repo / "convert_hf_to_gguf.py"), merged_dir,
          "--outfile", str(out), "--outtype", "f16"])
    return str(out)


def make_imatrix(repo: Path, f16: str, calibration: str, gguf_dir: Path) -> str:
    out = gguf_dir / "imatrix.dat"
    _run([_bin(repo, "llama-imatrix"), "-m", f16, "-f", calibration,
          "-o", str(out), "-ngl", "99"])
    return str(out)


def quantize(repo: Path, f16: str, imatrix: str | None, quant: str, gguf_dir: Path) -> str:
    out = gguf_dir / f"model-{quant}.gguf"
    cmd = [_bin(repo, "llama-quantize")]
    if imatrix and not quant.startswith("Q8"):
        cmd += ["--imatrix", imatrix]
    cmd += [f16, str(out), quant]
    _run(cmd)
    return str(out)


def file_size_mb(path: str) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def run_full_conversion(cfg: Config, tokenizer, merged_dir: str, workdir) -> list[dict]:
    gguf_dir = cfg.path("gguf_dir")
    repo = build_llama_cpp(cfg, workdir)
    f16 = convert_to_f16(repo, merged_dir, gguf_dir)

    imatrix = None
    if cfg.gguf["use_imatrix"]:
        calib = write_calibration_file(cfg, tokenizer)
        imatrix = make_imatrix(repo, f16, calib, gguf_dir)

    rows = [{"label": "f16", "path": f16, "size_mb": file_size_mb(f16)}]
    for quant in cfg.gguf["quant_types"]:
        path = quantize(repo, f16, imatrix, quant, gguf_dir)
        rows.append({"label": quant, "path": path, "size_mb": file_size_mb(path)})
        print(f"[gguf] {quant}: {rows[-1]['size_mb']:.0f} MB")
    return rows

def save_gguf_unsloth(cfg: Config, model, tokenizer) -> None:
    quants = [q.lower() for q in cfg.gguf["quant_types"]]
    model.save_pretrained_gguf(
        str(cfg.path("gguf_dir")), tokenizer, quantization_method=quants
    )
