import torch

from model.tokenizer import CharTokenizer
from model.ctc_decoder import CTCDecoder


def _one_hot_log_probs(ids, vocab_size, high=10.0, low=-10.0):
    t = len(ids)
    logits = torch.full((t, vocab_size), low)
    for i, idx in enumerate(ids):
        logits[i, idx] = high
    return torch.log_softmax(logits, dim=-1)


def test_ctc_greedy_decode_collapses_repeats_and_removes_blanks():
    tokenizer = CharTokenizer.from_texts(["ab"])
    a_id = tokenizer.stoi["a"]
    b_id = tokenizer.stoi["b"]
    blank = tokenizer.blank_id

    # sequence: a, a, blank, b, b, b -> expected decoded text "ab"
    ids = [a_id, a_id, blank, b_id, b_id, b_id]
    log_probs = _one_hot_log_probs(ids, tokenizer.vocab_size)

    decoder = CTCDecoder(tokenizer)
    result = decoder.decode(log_probs)

    assert result.text == "ab"
    assert len(result.confidences) == 2
    for conf in result.confidences:
        assert 0.0 <= conf <= 1.0
        assert conf > 0.9  # near-one-hot logits should give high confidence


def test_ctc_decode_batch():
    tokenizer = CharTokenizer.from_texts(["ab"])
    a_id = tokenizer.stoi["a"]
    blank = tokenizer.blank_id

    ids = [a_id, blank]
    log_probs = _one_hot_log_probs(ids, tokenizer.vocab_size)  # (T, V)
    batch = log_probs.unsqueeze(1).repeat(1, 3, 1)  # (T, B=3, V)

    decoder = CTCDecoder(tokenizer)
    results = decoder.decode_batch(batch)

    assert len(results) == 3
    assert all(r.text == "a" for r in results)
