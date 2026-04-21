"""Tests for BPE tokenizer."""

from maayatrain.training.tokenizer import BPETokenizer


def test_train_and_encode():
    """Training produces a vocabulary and encoding works."""
    text = "hello world hello world hello world foo bar baz " * 50
    tok = BPETokenizer(vocab_size=100)
    tok.train(text)

    assert tok.vocab_size > 0
    assert tok.vocab_size <= 100

    ids = tok.encode("hello world")
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


def test_roundtrip():
    """Encoding then decoding recovers the original text."""
    text = "the quick brown fox jumps over the lazy dog " * 100
    tok = BPETokenizer(vocab_size=200)
    tok.train(text)

    test_str = "the quick brown fox"
    ids = tok.encode(test_str)
    decoded = tok.decode(ids)
    assert decoded == test_str


def test_special_tokens():
    """Special tokens are correctly assigned and handled."""
    tok = BPETokenizer(vocab_size=50)
    tok.train("hello world test " * 50)

    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.bos_id == 2
    assert tok.eos_id == 3

    ids = tok.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_unknown_tokens():
    """Unknown characters get the <unk> token ID."""
    tok = BPETokenizer(vocab_size=30)
    tok.train("aaabbbccc " * 100)

    # Encode text with characters not in training data
    ids = tok.encode("xyz")
    assert tok.unk_id in ids


def test_save_and_load(tmp_path):
    """Tokenizer state persists through save/load."""
    text = "hello world testing tokenizer " * 100
    tok = BPETokenizer(vocab_size=100)
    tok.train(text)

    path = tmp_path / "tokenizer.json"
    tok.save(path)

    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size

    # Same encoding
    test_str = "hello world"
    assert tok.encode(test_str) == loaded.encode(test_str)


def test_empty_text():
    """Encoding empty text returns empty list."""
    tok = BPETokenizer(vocab_size=50)
    tok.train("hello world " * 50)
    assert tok.encode("") == []


def test_bos_eos_only():
    """With add_bos/add_eos on empty text, get only special tokens."""
    tok = BPETokenizer(vocab_size=50)
    tok.train("hello world " * 50)
    ids = tok.encode("", add_bos=True, add_eos=True)
    assert ids == [tok.bos_id, tok.eos_id]
