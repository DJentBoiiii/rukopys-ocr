"""Two-stage training entrypoint: pretrain on synthetic data, then finetune on RUKOPYS train."""
from __future__ import annotations

import argparse
import multiprocessing
import time
from pathlib import Path

import jiwer
import torch
from torch.utils.data import DataLoader

from model.crnn import CRNNModel
from model.ctc_decoder import CTCDecoder
from model.tokenizer import CharTokenizer
from training import config
from training.datasets_common import LineImageDataset, collate_batch
from training.evaluate import predict as ocr_predict
from training.rukopys_loader import (
    _load_split_regions,
    build_or_load_val_ids,
    download_rukopys,
    load_rukopys_train_subset,
    load_rukopys_val_set,
)
from training.synthetic_loader import load_synthetic_subset

LOG_FILE = config.LOG_DIR / "train.log"


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def build_or_load_vocab(synthetic_samples) -> CharTokenizer:
    if config.VOCAB_PATH.exists():
        return CharTokenizer.load(config.VOCAB_PATH)

    texts = [text for _, text in synthetic_samples]
    train_regions = _load_split_regions(config.DATA_CACHE_DIR, "train")
    texts += [r["text"] for r in train_regions]

    tokenizer = CharTokenizer.from_texts(texts)
    tokenizer.save(config.VOCAB_PATH)
    return tokenizer


def make_model(vocab_size: int) -> CRNNModel:
    return CRNNModel(
        vocab_size=vocab_size,
        img_height=config.MODEL_IMG_HEIGHT,
        cnn_channels=config.MODEL_CNN_CHANNELS,
        rnn_hidden=config.MODEL_RNN_HIDDEN,
        rnn_layers=config.MODEL_RNN_LAYERS,
    )


def save_checkpoint(model: CRNNModel, path, epoch: int, step: int) -> None:
    torch.save({"model_state": model.state_dict(), "epoch": epoch, "step": step}, path)


def evaluate_on_subset(model, tokenizer, decoder, val_samples) -> tuple[float, float]:
    """Compute WER/CER of `model` on a small preloaded validation subset
    (list of {"image", "text"} dicts), reusing the inference logic from
    `training/evaluate.py` instead of duplicating it.
    """
    model.eval()
    refs = [sample["text"] for sample in val_samples]
    hyps = [ocr_predict(model, decoder, sample["image"]) for sample in val_samples]
    model.train()
    return jiwer.wer(refs, hyps), jiwer.cer(refs, hyps)


def train_loop(
    model, dataset, epochs, batch_size, lr, checkpoint_path, stage_name,
    tokenizer=None, val_samples=None,
) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=6,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    # validation WER/CER tracking is only meaningful for the finetune stage
    # (real handwriting), not for pretrain on synthetic data
    track_val = stage_name == "finetune" and val_samples and tokenizer is not None
    decoder = CTCDecoder(tokenizer) if track_val else None

    checkpoint_path = Path(checkpoint_path)
    best_checkpoint_path = checkpoint_path.with_name(checkpoint_path.stem + "_best" + checkpoint_path.suffix)
    best_val_cer = float("inf")

    step = 0
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for images, targets, input_lengths, target_lengths in loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            log_probs = model(images)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            if step % config.CHECKPOINT_EVERY_STEPS == 0:
                save_checkpoint(model, checkpoint_path, epoch, step)
                log(f"[{stage_name}] step {step} epoch {epoch} loss {loss.item():.4f} (checkpoint saved)")

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - start_time
        save_checkpoint(model, checkpoint_path, epoch, step)

        if track_val:
            val_wer, val_cer = evaluate_on_subset(model, tokenizer, decoder, val_samples)
            log(
                f"[{stage_name}] epoch {epoch}/{epochs} avg_loss {avg_loss:.4f} "
                f"val_wer {val_wer:.4f} val_cer {val_cer:.4f} (checkpoint saved)"
            )

            if val_cer < best_val_cer:
                best_val_cer = val_cer
                save_checkpoint(model, best_checkpoint_path, epoch, step)
                log(f"new best val_cer: {val_cer:.4f} (epoch {epoch})")
        else:
            log(f"[{stage_name}] epoch {epoch}/{epochs} avg_loss {avg_loss:.4f} elapsed {elapsed / 60:.1f}min")

    return step


def run_pretrain() -> None:
    log("=== pretrain stage start ===")
    t0 = time.time()

    download_rukopys(str(config.DATA_CACHE_DIR))  # needed to build the combined vocab

    synthetic_samples = load_synthetic_subset(config.PRETRAIN_N_EXAMPLES)
    log(f"loaded {len(synthetic_samples)} synthetic examples")

    tokenizer = build_or_load_vocab(synthetic_samples)
    log(f"vocab size {tokenizer.vocab_size}")

    dataset = LineImageDataset(synthetic_samples, tokenizer)
    model = make_model(tokenizer.vocab_size)

    train_loop(
        model, dataset, config.PRETRAIN_EPOCHS, config.PRETRAIN_BATCH_SIZE,
        config.PRETRAIN_LR, config.PRETRAIN_CHECKPOINT, "pretrain",
    )

    elapsed = time.time() - t0
    log(f"=== pretrain stage done in {elapsed / 60:.1f} min ===")


def run_finetune() -> None:
    log("=== finetune stage start ===")
    t0 = time.time()

    download_rukopys(str(config.DATA_CACHE_DIR))

    tokenizer = CharTokenizer.load(config.VOCAB_PATH)
    val_ids = set(build_or_load_val_ids(config.DATA_CACHE_DIR))
    log(f"validation split: {len(val_ids)} regions (excluded from training)")

    samples = load_rukopys_train_subset(
        config.DATA_CACHE_DIR, config.FINETUNE_MAX_EXAMPLES, exclude_ids=val_ids,
    )
    log(f"loaded {len(samples)} rukopys train examples")

    val_samples = load_rukopys_val_set(config.DATA_CACHE_DIR)
    log(f"loaded {len(val_samples)} validation examples")

    dataset = LineImageDataset(samples, tokenizer)
    model = make_model(tokenizer.vocab_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.FINETUNE_CHECKPOINT.exists():
        state = torch.load(config.FINETUNE_CHECKPOINT, map_location=device)
        model.load_state_dict(state["model_state"])
        model = model.to(device)
        log("loaded finetune checkpoint weights (warm-start from previous finetune run)")
    elif config.PRETRAIN_CHECKPOINT.exists():
        state = torch.load(config.PRETRAIN_CHECKPOINT, map_location=device)
        model.load_state_dict(state["model_state"])
        model = model.to(device)
        log("loaded pretrain checkpoint weights")
    else:
        log("no checkpoint found, finetuning from scratch")

    train_loop(
        model, dataset, config.FINETUNE_EPOCHS, config.FINETUNE_BATCH_SIZE,
        config.FINETUNE_LR, config.FINETUNE_CHECKPOINT_GPU, "finetune",
        tokenizer=tokenizer, val_samples=val_samples,
    )

    elapsed = time.time() - t0
    log(f"=== finetune stage done in {elapsed / 60:.1f} min ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    args = parser.parse_args()

    torch.set_num_threads(multiprocessing.cpu_count())

    if args.stage == "pretrain":
        run_pretrain()
    else:
        run_finetune()


if __name__ == "__main__":
    main()
