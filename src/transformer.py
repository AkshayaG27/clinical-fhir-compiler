"""
Full Decoder-Only Clinical Transformer Architecture from Scratch.
Implements Pre-LN Decoder blocks with RMSNorm, RoPE Multi-Head Attention, and SwiGLU.
Supports both parallel training and KV-cached autoregressive generation.
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Callable
from src.layers import RMSNorm, SwiGLU
from src.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer Decoder Block:
    x_mid = x + CausalSelfAttention(RMSNorm(x))
    x_out = x_mid + SwiGLU(RMSNorm(x_mid))
    """

    def __init__(self, dim: int, n_heads: int, hidden_dim: int, max_seq_len: int = 128):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim=dim, n_heads=n_heads, max_seq_len=max_seq_len)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim=dim, hidden_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN Attention
        normed_attn = self.attn_norm(x)
        attn_out, new_cache = self.attn(normed_attn, start_pos=start_pos, kv_cache=kv_cache)
        x = x + attn_out

        # Pre-LN SwiGLU FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class ClinicalTransformerDecoder(nn.Module):
    """
    Scratch PyTorch Decoder-Only Transformer for Clinical Shorthand Translation.
    Edge-optimized parameter configuration: ~800K parameters.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        hidden_dim: Optional[int] = None,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # SwiGLU standard expansion: 8/3 * dim rounded to multiple of 8
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim / 3))
            hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    hidden_dim=hidden_dim,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = RMSNorm(dim)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights between embedding and output projection
        self.output_head.weight = self.tok_embeddings.weight

        self._init_weights()

    def _init_weights(self):
        """Initializes weights with normal distribution scaled by 1/sqrt(2*layers)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_kv_cache(
        self, batch_size: int, device: torch.device
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Pre-allocates contiguous tensor buffers for all layers."""
        head_dim = self.dim // self.n_heads
        caches = []
        for _ in range(self.n_layers):
            k_buf = torch.zeros(
                batch_size, self.n_heads, self.max_seq_len, head_dim, device=device
            )
            v_buf = torch.zeros(
                batch_size, self.n_heads, self.max_seq_len, head_dim, device=device
            )
            caches.append((k_buf, v_buf))
        return caches

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass.
        Args:
            input_ids: Tensor of token indices (B, L)
            start_pos: Starting position index for RoPE & KV-cache
            kv_caches: Optional list of per-layer (cached_K, cached_V) tuples
        Returns:
            logits: (B, L, vocab_size)
            new_caches: Updated KV-cache list if kv_caches were passed
        """
        x = self.tok_embeddings(input_ids)

        new_caches = [] if kv_caches is not None else None

        for idx, layer in enumerate(self.layers):
            cache_i = kv_caches[idx] if kv_caches is not None else None
            x, updated_cache_i = layer(x, start_pos=start_pos, kv_cache=cache_i)
            if new_caches is not None:
                new_caches.append(updated_cache_i)

        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits, new_caches

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 64,
        eos_token_id: int = 2,
        dfa_mask_fn: Optional[Callable[[List[int]], torch.Tensor]] = None,
        temperature: float = 1.0,
        use_cache: bool = True,
    ) -> List[int]:
        """
        Autoregressive generation with KV-cache and optional DFA logit masking.
        prompt_tokens: (1, seq_len)
        """
        device = prompt_tokens.device
        generated = prompt_tokens.squeeze(0).tolist()

        if use_cache:
            # Prefill phase: feed prompt into KV-cache
            kv_caches = self.init_kv_cache(batch_size=1, device=device)
            prompt_len = prompt_tokens.shape[1]
            logits, kv_caches = self.forward(
                prompt_tokens, start_pos=0, kv_caches=kv_caches
            )
            next_token_logits = logits[0, -1, :].clone()

            cur_pos = prompt_len
            for _ in range(max_new_tokens):
                # Apply DFA logit mask if provided
                if dfa_mask_fn is not None:
                    mask = dfa_mask_fn(generated).to(device)
                    next_token_logits = next_token_logits + mask

                if temperature > 0.0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).item()

                generated.append(next_token)
                if next_token == eos_token_id or cur_pos >= self.max_seq_len - 1:
                    break

                # Single-token decode step: attend query to cached keys/values
                next_input = torch.tensor([[next_token]], device=device)
                step_logits, kv_caches = self.forward(
                    next_input, start_pos=cur_pos, kv_caches=kv_caches
                )
                next_token_logits = step_logits[0, -1, :].clone()
                cur_pos += 1
        else:
            # Naive non-cached generation for complexity comparison
            for _ in range(max_new_tokens):
                cur_input = torch.tensor([generated], device=device)
                logits, _ = self.forward(cur_input, start_pos=0, kv_caches=None)
                next_token_logits = logits[0, -1, :].clone()

                if dfa_mask_fn is not None:
                    mask = dfa_mask_fn(generated).to(device)
                    next_token_logits = next_token_logits + mask

                if temperature > 0.0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).item()

                generated.append(next_token)
                if next_token == eos_token_id or len(generated) >= self.max_seq_len:
                    break

        return generated
