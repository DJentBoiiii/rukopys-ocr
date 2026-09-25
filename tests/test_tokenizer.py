import tempfile
from pathlib import Path

from model.tokenizer import CharTokenizer, BLANK_TOKEN


def test_tokenizer_roundtrip_encode_decode():
    texts = ["привіт світ", "Архів ЦДАВО 1919-1935"]
    tokenizer = CharTokenizer.from_texts(texts)

    assert tokenizer.itos[0] == BLANK_TOKEN
    assert tokenizer.blank_id == 0

    for text in texts:
        ids = tokenizer.encode(text)
        assert tokenizer.blank_id not in ids
        decoded = tokenizer.decode(ids)
        assert decoded == text


def test_tokenizer_save_and_load_roundtrip():
    tokenizer = CharTokenizer.from_texts(["абвгд"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vocab.json"
        tokenizer.save(path)
        loaded = CharTokenizer.load(path)

    assert loaded.itos == tokenizer.itos
    assert loaded.encode("абвгд") == tokenizer.encode("абвгд")


def test_tokenizer_unknown_char_raises():
    tokenizer = CharTokenizer.from_texts(["абв"])
    try:
        tokenizer.encode("xyz")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown character")
