"""BPE Tokenizer for MaayaTrain.

Replaces character-level tokenization with byte-pair encoding for
production-quality training. Independently implemented from the
BPE algorithm description (Sennrich et al., 2016 — public paper).

Features:
- Train a BPE vocabulary from raw text
- Save/load tokenizer state (JSON)
- Encode text → token IDs
- Decode token IDs → text
- Special tokens: <pad>, <unk>, <bos>, <eos>

Author: Akhil Ageer — MaayaTrain project
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("maayatrain.tokenizer")


# Pre-tokenization regex: split on whitespace boundaries while preserving spaces
_PRETOKENIZE_RE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+""")


class BPETokenizer:
    """Byte-Pair Encoding tokenizer for text.

    Parameters
    ----------
    vocab_size : int
        Target vocabulary size (including special tokens).
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, vocab_size: int = 4096) -> None:
        self.target_vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self._trained = False

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.vocab[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.EOS_TOKEN]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, *, verbose: bool = False) -> None:
        """Train the BPE tokenizer on raw text.

        Algorithm (Sennrich et al., 2016):
        1. Initialize vocabulary with all unique bytes/characters
        2. Count all adjacent pair frequencies
        3. Merge the most frequent pair into a new token
        4. Repeat until target vocab_size is reached

        Parameters
        ----------
        text : str
            Raw training text.
        verbose : bool
            Log merge operations.
        """
        # Pre-tokenize: split into words
        words = _PRETOKENIZE_RE.findall(text)

        # Initialize: each word → list of characters, with frequency count
        word_freqs: Dict[Tuple[str, ...], int] = Counter()
        for word in words:
            chars = tuple(word)
            word_freqs[chars] += 1

        # Build initial character vocabulary
        chars = set()
        for word_tuple in word_freqs:
            chars.update(word_tuple)

        # Start with special tokens + characters
        self.vocab = {}
        for i, special in enumerate(self.SPECIAL_TOKENS):
            self.vocab[special] = i
        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        # Iteratively merge most frequent pairs
        num_merges = self.target_vocab_size - len(self.vocab)
        self.merges = []

        for merge_idx in range(max(0, num_merges)):
            # Count pair frequencies
            pair_counts: Counter = Counter()
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Find best pair
            best_pair = pair_counts.most_common(1)[0][0]
            best_count = pair_counts[best_pair]

            if best_count < 2:
                break  # No more useful merges

            # Merge the pair in all words
            merged_token = best_pair[0] + best_pair[1]
            new_word_freqs: Dict[Tuple[str, ...], int] = {}

            for word_tuple, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word_tuple):
                    if (
                        i < len(word_tuple) - 1
                        and word_tuple[i] == best_pair[0]
                        and word_tuple[i + 1] == best_pair[1]
                    ):
                        new_word.append(merged_token)
                        i += 2
                    else:
                        new_word.append(word_tuple[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq

            word_freqs = new_word_freqs
            self.merges.append(best_pair)
            self.vocab[merged_token] = len(self.vocab)

            if verbose and (merge_idx + 1) % 500 == 0:
                logger.info(
                    "BPE merge %d/%d: '%s' + '%s' → '%s' (count=%d)",
                    merge_idx + 1,
                    num_merges,
                    best_pair[0],
                    best_pair[1],
                    merged_token,
                    best_count,
                )

        # Build inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._trained = True

        logger.info(
            "BPE training complete: vocab_size=%d, merges=%d",
            len(self.vocab),
            len(self.merges),
        )

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to a list of token IDs.

        Parameters
        ----------
        text : str
            Input text.
        add_bos : bool
            Prepend <bos> token.
        add_eos : bool
            Append <eos> token.

        Returns
        -------
        list[int]
            Token IDs.
        """
        tokens: List[int] = []

        if add_bos:
            tokens.append(self.bos_id)

        # Pre-tokenize
        words = _PRETOKENIZE_RE.findall(text)

        for word in words:
            # Start with characters
            word_tokens = list(word)

            # Apply merges in order
            for pair in self.merges:
                i = 0
                new_tokens = []
                while i < len(word_tokens):
                    if (
                        i < len(word_tokens) - 1
                        and word_tokens[i] == pair[0]
                        and word_tokens[i + 1] == pair[1]
                    ):
                        new_tokens.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens

            # Map to IDs
            for token in word_tokens:
                tokens.append(self.vocab.get(token, self.unk_id))

        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text.

        Parameters
        ----------
        ids : list[int]
            Token IDs to decode.

        Returns
        -------
        str
            Decoded text. Special tokens are omitted.
        """
        parts = []
        for token_id in ids:
            token = self.inverse_vocab.get(token_id, "")
            if token in self.SPECIAL_TOKENS:
                continue
            parts.append(token)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save tokenizer state to a JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
            "target_vocab_size": self.target_vocab_size,
            "version": "1.0",
            "author": "Akhil Ageer — MaayaTrain",
        }
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Tokenizer saved to %s (vocab_size=%d)", path, len(self.vocab))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer from a JSON file.

        Parameters
        ----------
        path : str or Path
            Path to the saved tokenizer.

        Returns
        -------
        BPETokenizer
            Loaded tokenizer ready for encode/decode.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(vocab_size=data.get("target_vocab_size", 4096))
        tok.vocab = data["vocab"]
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        tok._trained = True
        logger.info("Tokenizer loaded from %s (vocab_size=%d)", path, len(tok.vocab))
        return tok
