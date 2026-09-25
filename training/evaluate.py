"""Evaluate the trained CRNN on the official RUKOPYS test split (WER/CER),
and compare against a pytesseract baseline on the same line crops.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jiwer
import torch
from PIL import Image

from model.crnn import CRNNModel
from model.ctc_decoder import CTCDecoder
from model.reconstruction import ReconstructionModule, build_word_frequencies
from model.tokenizer import CharTokenizer
from training import config
from training.datasets_common import normalize_image, resize_to_fixed_height
from training.rukopys_loader import (
    _load_split_regions,
    _region_id,
    build_or_load_val_ids,
    download_rukopys,
    load_rukopys_test_set,
)

try:
    import pytesseract
except ImportError:
    pytesseract = None

RESULTS_JSON = config.PROJECT_ROOT / "experiments" / "eval_results.json"


def load_model_for_eval() -> tuple[CRNNModel, CharTokenizer, CTCDecoder]:
    tokenizer = CharTokenizer.load(config.VOCAB_PATH)
    model = CRNNModel(
        vocab_size=tokenizer.vocab_size,
        img_height=config.MODEL_IMG_HEIGHT,
        cnn_channels=config.MODEL_CNN_CHANNELS,
        rnn_hidden=config.MODEL_RNN_HIDDEN,
        rnn_layers=config.MODEL_RNN_LAYERS,
    )
    checkpoint_path = config.FINETUNE_CHECKPOINT if config.FINETUNE_CHECKPOINT.exists() else config.PRETRAIN_CHECKPOINT
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()
    decoder = CTCDecoder(tokenizer)
    return model, tokenizer, decoder


def predict(model: CRNNModel, decoder: CTCDecoder, image) -> str:
    resized = resize_to_fixed_height(image)
    tensor = normalize_image(resized).unsqueeze(0)  # (1, 1, H, W)
    with torch.no_grad():
        log_probs = model(tensor)  # (T, 1, V)
    return decoder.decode(log_probs[:, 0, :]).text


def predict_tesseract(image) -> str | None:
    if pytesseract is None:
        return None
    try:
        pil_image = Image.fromarray(image)
        return pytesseract.image_to_string(pil_image, lang="ukr").strip()
    except Exception:
        return None


def build_reconstruction_module() -> ReconstructionModule:
    """Build the word-frequency dictionary from RUKOPYS train transcriptions,
    excluding the fixed validation subset (see `training/rukopys_loader.py`)
    so it stays disjoint from any WER/CER-tracked data.
    """
    val_ids = set(build_or_load_val_ids(config.DATA_CACHE_DIR))
    regions = _load_split_regions(config.DATA_CACHE_DIR, "train")
    texts = [r["text"] for r in regions if _region_id(r) not in val_ids]
    freqs = build_word_frequencies(texts)
    return ReconstructionModule(freqs)


def evaluate() -> None:
    download_rukopys(str(config.DATA_CACHE_DIR))
    test_samples = load_rukopys_test_set(config.DATA_CACHE_DIR)
    print(f"loaded {len(test_samples)} test regions")

    model, tokenizer, decoder = load_model_for_eval()
    reconstructor = build_reconstruction_module()

    refs, hyps_model, hyps_recon, hyps_tesseract = [], [], [], []
    per_source = {}

    t0 = time.time()
    tesseract_available = pytesseract is not None
    for i, sample in enumerate(test_samples):
        ref = sample["text"]
        hyp_model = predict(model, decoder, sample["image"])
        hyp_recon = reconstructor.reconstruct(hyp_model)

        refs.append(ref)
        hyps_model.append(hyp_model)
        hyps_recon.append(hyp_recon)
        per_source.setdefault(
            sample["source"], {"refs": [], "model": [], "recon": [], "tesseract": []},
        )
        per_source[sample["source"]]["refs"].append(ref)
        per_source[sample["source"]]["model"].append(hyp_model)
        per_source[sample["source"]]["recon"].append(hyp_recon)

        if tesseract_available:
            hyp_tess = predict_tesseract(sample["image"])
            if hyp_tess is None:
                tesseract_available = False
                print("tesseract not usable in this environment, skipping tesseract comparison")
            else:
                hyps_tesseract.append(hyp_tess)
                per_source[sample["source"]]["tesseract"].append(hyp_tess)

        if (i + 1) % 50 == 0:
            print(f"  evaluated {i + 1}/{len(test_samples)}")

    elapsed = time.time() - t0

    results = {
        "n_examples": len(test_samples),
        "model_wer": jiwer.wer(refs, hyps_model),
        "model_cer": jiwer.cer(refs, hyps_model),
        "reconstruction_wer": jiwer.wer(refs, hyps_recon),
        "reconstruction_cer": jiwer.cer(refs, hyps_recon),
        "eval_time_seconds": elapsed,
    }

    if hyps_tesseract and len(hyps_tesseract) == len(refs):
        results["tesseract_wer"] = jiwer.wer(refs, hyps_tesseract)
        results["tesseract_cer"] = jiwer.cer(refs, hyps_tesseract)
    else:
        results["tesseract_wer"] = None
        results["tesseract_cer"] = None

    results["by_source"] = {}
    for source, data in per_source.items():
        entry = {
            "n": len(data["refs"]),
            "model_wer": jiwer.wer(data["refs"], data["model"]),
            "model_cer": jiwer.cer(data["refs"], data["model"]),
            "reconstruction_wer": jiwer.wer(data["refs"], data["recon"]),
            "reconstruction_cer": jiwer.cer(data["refs"], data["recon"]),
        }
        if data["tesseract"] and len(data["tesseract"]) == len(data["refs"]):
            entry["tesseract_wer"] = jiwer.wer(data["refs"], data["tesseract"])
            entry["tesseract_cer"] = jiwer.cer(data["refs"], data["tesseract"])
        results["by_source"][source] = entry

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    evaluate()
