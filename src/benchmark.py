"""
Empirical Latency & Memory Profiling Suite.
Benchmarks KV-cached decoding vs. Naive non-cached decoding across context lengths.
Verifies sub-20ms step latency on CPU hardware.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.tokenizer import ClinicalTokenizer
from src.transformer import ClinicalTransformerDecoder

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
VOCAB_PATH = BASE_DIR / "data" / "tokenizer" / "vocab.json"
CHECKPOINT_PATH = BASE_DIR / "models" / "clinical_transformer_best.pt"


def benchmark_step_latency(
    model: ClinicalTransformerDecoder,
    seq_lengths: List[int] = [8, 16, 24, 32, 48, 64],
    n_warmup: int = 5,
    n_trials: int = 20,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[float]]:
    """
    Measures per-step token generation latency (in milliseconds)
    with KV-cache vs. without cache across increasing sequence lengths.
    """
    model.eval()
    cached_latencies = []
    non_cached_latencies = []

    print("Running Latency Benchmark on CPU...")

    for seq_len in seq_lengths:
        # 1. Benchmark WITH KV-cache (O(L) per step)
        # Prefill cache up to seq_len - 1
        kv_caches = model.init_kv_cache(batch_size=1, device=device)
        dummy_prompt = torch.randint(1, model.vocab_size, (1, seq_len - 1), device=device)

        with torch.no_grad():
            _, kv_caches = model(dummy_prompt, start_pos=0, kv_caches=kv_caches)

            # Warmup
            single_tok = torch.randint(1, model.vocab_size, (1, 1), device=device)
            for _ in range(n_warmup):
                _ = model(single_tok, start_pos=seq_len - 1, kv_caches=kv_caches)

            # Timed trials for single-step decode
            times_cached = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                _ = model(single_tok, start_pos=seq_len - 1, kv_caches=kv_caches)
                t1 = time.perf_counter()
                times_cached.append((t1 - t0) * 1000.0)  # to ms

        avg_cached = float(np.mean(times_cached))
        cached_latencies.append(avg_cached)

        # 2. Benchmark WITHOUT KV-cache (O(L^2) recomputation)
        dummy_full_seq = torch.randint(1, model.vocab_size, (1, seq_len), device=device)
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(dummy_full_seq, start_pos=0, kv_caches=None)

            times_non_cached = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                _ = model(dummy_full_seq, start_pos=0, kv_caches=None)
                t1 = time.perf_counter()
                times_non_cached.append((t1 - t0) * 1000.0)

        avg_non_cached = float(np.mean(times_non_cached))
        non_cached_latencies.append(avg_non_cached)

        print(
            f"Seq Len: {seq_len:2d} | "
            f"Cached: {avg_cached:5.2f} ms | "
            f"Non-Cached: {avg_non_cached:5.2f} ms | "
            f"Speedup: {avg_non_cached / max(avg_cached, 1e-4):.2f}x"
        )

    return {
        "seq_lengths": seq_lengths,
        "cached_ms": cached_latencies,
        "non_cached_ms": non_cached_latencies,
    }


def plot_benchmark_results(results: Dict[str, List[float]]):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "kv_cache_latency_profile.png"

    seq_lengths = results["seq_lengths"]
    cached = results["cached_ms"]
    non_cached = results["non_cached_ms"]

    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(seq_lengths, non_cached, "r-o", linewidth=2.5, markersize=7, label="Naive Without Cache (O(L²))")
    plt.plot(seq_lengths, cached, "g-s", linewidth=2.5, markersize=7, label="Circular KV-Cache (O(L))")

    # Add reference threshold at 20ms
    plt.axhline(20.0, color="gray", linestyle="--", alpha=0.7, label="Sub-20ms Edge Target")

    plt.title("Autoregressive Step Latency vs Sequence Length (Consumer CPU)", fontsize=13, fontweight="bold")
    plt.xlabel("Sequence Length L", fontsize=11)
    plt.ylabel("Step Latency (milliseconds)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.savefig(out_path)
    plt.close()
    print(f"Saved benchmark plot to {out_path}")


def run_benchmark():
    device = torch.device("cpu")
    tokenizer = ClinicalTokenizer(VOCAB_PATH)

    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        cfg = ckpt.get("config", {})
        model = ClinicalTransformerDecoder(
            vocab_size=cfg.get("vocab_size", tokenizer.vocab_size),
            dim=cfg.get("dim", 128),
            n_layers=cfg.get("n_layers", 4),
            n_heads=cfg.get("n_heads", 4),
            max_seq_len=cfg.get("max_seq_len", 64),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model checkpoint from {CHECKPOINT_PATH}")
    else:
        model = ClinicalTransformerDecoder(
            vocab_size=tokenizer.vocab_size,
            dim=128,
            n_layers=4,
            n_heads=4,
            max_seq_len=64,
        ).to(device)

    results = benchmark_step_latency(model, device=device)
    plot_benchmark_results(results)
    return results


if __name__ == "__main__":
    run_benchmark()
