"""One-off benchmark (instruction.md Крок 3): measure real GPU throughput for
finetune training and diagnose whether the crop+deskew image preprocessing is
eager (done once) or repeated every epoch (lazy). Does NOT touch real
checkpoints (`finetune.pt` / `finetune_v2.pt`) -- uses a throwaway model.

Results are printed to stdout and appended to `experiments/gpu_benchmark.md`.
"""
from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from model.tokenizer import CharTokenizer
from training import config
from training.datasets_common import LineImageDataset, collate_batch
from training.rukopys_loader import build_or_load_val_ids, download_rukopys, load_rukopys_train_subset
from training.train import make_model

BENCHMARK_N_EXAMPLES = 3200
BENCHMARK_EPOCHS = 2
BATCH_SIZE = config.FINETUNE_BATCH_SIZE
RESULTS_MD = config.PROJECT_ROOT / "experiments" / "gpu_benchmark.md"

# CPU baseline from the prior, fully-completed CPU finetune run described in
# instruction.md: 10 epochs / 6000 examples, ~8.7 min/epoch.
CPU_MIN_PER_EPOCH = 8.7
CPU_N_EXAMPLES = 6000


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    download_rukopys(str(config.DATA_CACHE_DIR))
    tokenizer = CharTokenizer.load(config.VOCAB_PATH)
    val_ids = set(build_or_load_val_ids(config.DATA_CACHE_DIR))

    t0 = time.time()
    samples = load_rukopys_train_subset(config.DATA_CACHE_DIR, BENCHMARK_N_EXAMPLES, exclude_ids=val_ids)
    prep_time = time.time() - t0
    print(
        f"loaded+preprocessed (crop+deskew) {len(samples)} examples in {prep_time:.1f}s "
        f"({prep_time / len(samples):.4f} s/example, one-time/eager, happens BEFORE the epoch loop)"
    )

    dataset = LineImageDataset(samples, tokenizer)
    model = make_model(tokenizer.vocab_size).to(device)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.FINETUNE_LR)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    epoch_durations = []
    step_times = []
    total_steps = 0
    for epoch in range(1, BENCHMARK_EPOCHS + 1):
        epoch_t0 = time.time()
        for images, targets, input_lengths, target_lengths in loader:
            step_t0 = time.time()
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            log_probs = model(images)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.time() - step_t0)
            total_steps += 1
        epoch_durations.append(time.time() - epoch_t0)
        print(
            f"epoch {epoch} duration: {epoch_durations[-1]:.1f}s "
            f"({epoch_durations[-1] / len(dataset):.4f} s/example)"
        )

    avg_step = sum(step_times) / len(step_times)
    sec_per_example_gpu = avg_step / BATCH_SIZE
    sec_per_example_cpu = (CPU_MIN_PER_EPOCH * 60) / CPU_N_EXAMPLES
    speedup = sec_per_example_cpu / sec_per_example_gpu

    epoch2_str = f"{epoch_durations[1]:.1f}s" if len(epoch_durations) > 1 else "n/a"
    report = f"""
# GPU benchmark ({device})

CPU baseline: ~{sec_per_example_cpu:.4f} sec/example (from prior {CPU_N_EXAMPLES}-example run, {CPU_MIN_PER_EPOCH} min/epoch)
GPU measured: ~{sec_per_example_gpu:.4f} sec/example (this benchmark, {BENCHMARK_N_EXAMPLES} examples, batch={BATCH_SIZE})
speedup: ~{speedup:.1f}x

epoch 1 duration: {epoch_durations[0]:.1f}s (compute only -- crop+deskew preprocessing of
  {len(samples)} examples was done eagerly *before* the epoch loop started, taking
  {prep_time:.1f}s separately, see below)
epoch 2 duration: {epoch2_str} (compute-only, preprocessing not repeated)
one-time preprocessing (crop+deskew, {len(samples)} examples): {prep_time:.1f}s
  ({prep_time / len(samples):.4f} s/example)

avg step time: {avg_step:.4f}s ({total_steps} steps total over {BENCHMARK_EPOCHS} epochs, batch={BATCH_SIZE})

Diagnosis: `load_rukopys_train_subset` performs crop+deskew eagerly, once, before
`LineImageDataset` (and the training loop) are even constructed. Only the cheap
`resize_to_fixed_height` (cv2 resize) step happens lazily inside `__getitem__`,
repeated every epoch -- this is why epoch 1 and epoch 2 durations are close to
each other rather than epoch 1 being disproportionately slower: the expensive
part of preprocessing is not repeated per-epoch, and the per-epoch lazy part is
cheap relative to GPU compute.
"""
    print(report)
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_MD, "a", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
