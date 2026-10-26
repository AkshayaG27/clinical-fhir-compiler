"""
Domain Clinical Tokenizer.
Tokenizes doctor shorthand and compact IR with guaranteed roundtrip fidelity.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

BASE_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = BASE_DIR / "data" / "tokenizer"


class ClinicalTokenizer:
    """
    Regex-based domain tokenizer for clinical prescription translation.
    Treats medical keywords, drugs, dosages, frequencies, and delimiters as discrete tokens.
    """

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]
    PAD_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    SEP_TOKEN_ID = 3
    UNK_TOKEN_ID = 4

    # Regex splits words, dosages, digits, punctuation, and whitespace/newlines
    TOKEN_REGEX = re.compile(
        r"[A-Za-z0-9_.]+(?:mg|mcg|IU|g|ml|d|w)?|[A-Za-z0-9_.]+|[|\n,;+/:=-]"
    )

    def __init__(self, vocab_file: Optional[Union[str, Path]] = None):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        if vocab_file and Path(vocab_file).exists():
            self.load(vocab_file)
        else:
            self._init_special_tokens()

    def _init_special_tokens(self):
        self.token2id = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        self.id2token = {idx: tok for idx, tok in enumerate(self.SPECIAL_TOKENS)}

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def tokenize(self, text: str) -> List[str]:
        """Splits raw text into a list of normalized token strings."""
        raw_tokens = self.TOKEN_REGEX.findall(text)
        tokens = []
        for t in raw_tokens:
            # Preserve uppercase for record types and common clinical codes
            if t in ["MED", "COND", "OD", "BD", "TID", "QID", "HS", "SOS", "STAT", "IU"]:
                tokens.append(t)
            elif t in ["|", "\n", ",", ";", "+", "/", ":", "=", "-"]:
                tokens.append(t)
            else:
                tokens.append(t.lower())
        return tokens

    def build_vocab(self, texts: List[str]):
        """Builds vocabulary from a corpus of texts."""
        self._init_special_tokens()
        next_id = len(self.SPECIAL_TOKENS)

        unique_tokens = set()
        for text in texts:
            unique_tokens.update(self.tokenize(text))

        # Sort for deterministic vocabulary ordering
        for tok in sorted(unique_tokens):
            if tok not in self.token2id:
                self.token2id[tok] = next_id
                self.id2token[next_id] = tok
                next_id += 1

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        pad_to_max: bool = False,
    ) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.token2id.get(tok, self.UNK_TOKEN_ID) for tok in tokens]

        if add_bos:
            ids = [self.BOS_TOKEN_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_TOKEN_ID]

        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
            elif pad_to_max and len(ids) < max_length:
                ids = ids + [self.PAD_TOKEN_ID] * (max_length - len(ids))

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for i in ids:
            if skip_special_tokens and i in [
                self.PAD_TOKEN_ID,
                self.BOS_TOKEN_ID,
                self.EOS_TOKEN_ID,
                self.SEP_TOKEN_ID,
            ]:
                continue
            tokens.append(self.id2token.get(i, "<unk>"))

        # Reconstruct string
        res = ""
        for i, tok in enumerate(tokens):
            if tok in ["|", "\n"]:
                res += tok
            elif i > 0 and tokens[i - 1] in ["|", "\n"]:
                res += tok
            elif i > 0 and tok in [",", ";", ":"]:
                res += tok
            else:
                if res and not res.endswith(("|", "\n", " ")):
                    res += " "
                res += tok
        return res

    def prepare_training_sample(
        self, shorthand: str, target_ir: str, max_length: int = 64
    ) -> Tuple[List[int], List[int]]:
        """
        Creates teacher-forcing training pairs:
        input:  [<bos>, prompt..., <sep>, target..., <eos>, <pad>...]
        target: [-100,  -100...,  -100, target..., <eos>, -100...]
        Loss is calculated strictly on the target tokens!
        """
        prompt_ids = self.encode(shorthand)
        target_ids = self.encode(target_ir)

        # Full sequence: <bos> + prompt + <sep> + target + <eos>
        full_input = (
            [self.BOS_TOKEN_ID]
            + prompt_ids
            + [self.SEP_TOKEN_ID]
            + target_ids
            + [self.EOS_TOKEN_ID]
        )

        prompt_prefix_len = 1 + len(prompt_ids) + 1  # bos + prompt + sep

        # Mask prompt prefix with -100 (PyTorch ignore_index)
        full_target = (
            [-100] * prompt_prefix_len
            + target_ids
            + [self.EOS_TOKEN_ID]
        )

        # Truncate or Pad
        if len(full_input) > max_length:
            full_input = full_input[:max_length]
            full_target = full_target[:max_length]
        else:
            pad_len = max_length - len(full_input)
            full_input = full_input + [self.PAD_TOKEN_ID] * pad_len
            full_target = full_target + [-100] * pad_len

        return full_input, full_target

    def save(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "token2id": self.token2id,
                    "id2token": {str(k): v for k, v in self.id2token.items()},
                },
                f,
                indent=2,
            )

    def load(self, filepath: Union[str, Path]):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.token2id = data["token2id"]
        self.id2token = {int(k): v for k, v in data["id2token"].items()}


def train_and_save_tokenizer():
    """Fits tokenizer on synthetic data and saves to disk."""
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASE_DIR / "data" / "synthetic" / "train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    all_texts = []
    for d in train_data:
        all_texts.append(d["shorthand"])
        all_texts.append(d["target_ir"])

    tok = ClinicalTokenizer()
    tok.build_vocab(all_texts)
    vocab_path = TOKENIZER_DIR / "vocab.json"
    tok.save(vocab_path)
    print(f"Built ClinicalTokenizer with {tok.vocab_size} tokens, saved to {vocab_path}")
    return tok


if __name__ == "__main__":
    train_and_save_tokenizer()
