"""
Training Engine for Scratch Clinical Transformer Decoder.
Trains on paired shorthand <-> DSL tuples with teacher forcing.
Tracks cross-entropy loss and DSL string Exact-Match (EM) accuracy.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import ClinicalTokenizer
from src.transformer import ClinicalTransformerDecoder

BASE_DIR = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = BASE_DIR / "data" / "synthetic"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
VOCAB_PATH = BASE_DIR / "data" / "tokenizer" / "vocab.json"


class ClinicalDataset(Dataset):
    """PyTorch Dataset returning (input_ids, target_ids)."""

    def __init__(self, data_file: Path, tokenizer: ClinicalTokenizer, max_len: int = 64):
        with open(data_file, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.samples = []
        for r in self.records:
            inp, tgt = self.tokenizer.prepare_training_sample(
                r["shorthand"], r["target_ir"], max_length=self.max_len
            )
            self.samples.append(
                (
                    torch.tensor(inp, dtype=torch.long),
                    torch.tensor(tgt, dtype=torch.long),
                    r["shorthand"],
                    r["target_ir"],
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        inp, tgt, raw_shorthand, raw_ir = self.samples[idx]
        return inp, tgt, raw_shorthand, raw_ir


def evaluate(
    model: ClinicalTransformerDecoder,
    val_loader: DataLoader,
    tokenizer: ClinicalTokenizer,
    device: torch.device,
    compute_exact_match: bool = True,
    max_eval_samples: int = 150,
) -> Tuple[float, float]:
    """Computes validation cross-entropy loss and exact-match string accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            inps, tgts, _, _ = batch
            inps = inps.to(device)
            tgts = tgts.to(device)

            logits, _ = model(inps)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_tgts = tgts[:, 1:].contiguous()

            loss = criterion(shift_logits.view(-1, model.vocab_size), shift_tgts.view(-1))
            valid_tokens = (shift_tgts != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_val_loss = total_loss / max(total_tokens, 1)

    # Compute Exact-Match Accuracy on subset
    exact_matches = 0
    total_evaluated = 0

    if compute_exact_match:
        val_dataset = val_loader.dataset
        eval_indices = list(range(min(len(val_dataset), max_eval_samples)))

        for idx in eval_indices:
            _, _, shorthand, ground_truth_ir = val_dataset[idx]
            prompt_tokens = (
                [tokenizer.BOS_TOKEN_ID]
                + tokenizer.encode(shorthand)
                + [tokenizer.SEP_TOKEN_ID]
            )
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

            gen_ids = model.generate(
                prompt_tensor,
                max_new_tokens=40,
                eos_token_id=tokenizer.EOS_TOKEN_ID,
                temperature=0.0,
                use_cache=True,
            )

            # Strip prompt prefix
            generated_target_ids = gen_ids[len(prompt_tokens) :]
            decoded_ir = tokenizer.decode(generated_target_ids).strip()

            # Normalize whitespace for exact match comparison
            norm_gen = "\n".join(
                [line.strip().replace(" ", "") for line in decoded_ir.split("\n") if line.strip()]
            )
            norm_gt = "\n".join(
                [line.strip().replace(" ", "") for line in ground_truth_ir.split("\n") if line.strip()]
            )

            if norm_gen == norm_gt:
                exact_matches += 1
            total_evaluated += 1

    em_accuracy = (exact_matches / total_evaluated * 100.0) if total_evaluated > 0 else 0.0
    return avg_val_loss, em_accuracy


def train_model(
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    max_seq_len: int = 64,
    dim: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    patience: int = 3,
    min_delta: float = 0.05,
    device_str: Optional[str] = None,
):
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"=== Starting Training on device: {device} (Max Epochs: {epochs}, Patience: {patience}) ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = ClinicalTokenizer(VOCAB_PATH)
    print(f"Loaded Tokenizer with {tokenizer.vocab_size} tokens")

    train_dataset = ClinicalDataset(SYNTHETIC_DIR / "train.json", tokenizer, max_len=max_seq_len)
    val_dataset = ClinicalDataset(SYNTHETIC_DIR / "val.json", tokenizer, max_len=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ClinicalTransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
    ).to(device)

    param_count = model.count_parameters()
    print(f"Model Architecture: {n_layers} layers, {n_heads} heads, dim={dim}")
    print(f"Trainable Parameters: {param_count:,} (~{param_count * 4 / (1024**2):.2f} MB FP32)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_em = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        tokens_seen = 0

        for inps, tgts, _, _ in train_loader:
            inps = inps.to(device)
            tgts = tgts.to(device)

            optimizer.zero_grad()
            logits, _ = model(inps)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_tgts = tgts[:, 1:].contiguous()

            loss = criterion(shift_logits.view(-1, model.vocab_size), shift_tgts.view(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            valid_tokens = (shift_tgts != -100).sum().item()
            total_train_loss += loss.item() * valid_tokens
            tokens_seen += valid_tokens

        avg_train_loss = total_train_loss / max(tokens_seen, 1)
        val_loss, em_acc = evaluate(model, val_loader, tokenizer, device, compute_exact_match=True)

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_duration:.1f}s] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Exact Match: {em_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_em_accuracy": em_acc,
            "duration_sec": epoch_duration,
        })

        # Track improvements and save best checkpoint
        is_best = False
        if em_acc > best_em + min_delta:
            best_em = em_acc
            patience_counter = 0
            is_best = True
        elif val_loss < best_val_loss - 1e-4 and best_em == 0.0:
            # During early warmup before exact match activates
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1
            print(f" -> No improvement in Val EM (Best: {best_em:.2f}%, Patience: {patience_counter}/{patience})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if is_best or em_acc >= best_em:
            if em_acc > best_em:
                best_em = em_acc
            ckpt_path = MODELS_DIR / "clinical_transformer_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "vocab_size": tokenizer.vocab_size,
                        "dim": dim,
                        "n_layers": n_layers,
                        "n_heads": n_heads,
                        "max_seq_len": max_seq_len,
                    },
                    "val_loss": val_loss,
                    "val_em_accuracy": em_acc,
                },
                ckpt_path,
            )
            print(f" -> Saved Best Model Checkpoint to {ckpt_path} (EM: {em_acc:.2f}%)")

        # Early stopping if target exact match is achieved (>98%)
        if em_acc >= 98.0:
            print(f"Target accuracy reached (EM: {em_acc:.2f}% >= 98.0%). Early stopping.")
            break

        # Early stopping if patience exhausted (guarded to allow initial convergence)
        if patience_counter >= patience and epoch >= 6:
            print(f"Validation EM plateaued for {patience} consecutive epochs. Early stopping at epoch {epoch}.")
            break

    total_training_time = time.time() - start_time
    print(f"=== Training Completed in {total_training_time:.1f}s ({total_training_time/60:.2f} mins). Best EM: {best_em:.2f}% ===")

    # Save metrics history
    with open(REPORTS_DIR / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    train_model(epochs=15, batch_size=64, lr=1.5e-3, patience=3)
