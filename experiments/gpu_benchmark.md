
# GPU benchmark (cuda)

CPU baseline: ~0.0870 sec/example (from prior 6000-example run, 8.7 min/epoch)
GPU measured: ~0.0026 sec/example (this benchmark, 3200 examples, batch=16)
speedup: ~33.5x

epoch 1 duration: 9.6s (compute only -- crop+deskew preprocessing of
  3200 examples was done eagerly *before* the epoch loop started, taking
  135.8s separately, see below)
epoch 2 duration: 8.9s (compute-only, preprocessing not repeated)
one-time preprocessing (crop+deskew, 3200 examples): 135.8s
  (0.0424 s/example)

avg step time: 0.0415s (400 steps total over 2 epochs, batch=16)

Diagnosis: `load_rukopys_train_subset` performs crop+deskew eagerly, once, before
`LineImageDataset` (and the training loop) are even constructed. Only the cheap
`resize_to_fixed_height` (cv2 resize) step happens lazily inside `__getitem__`,
repeated every epoch -- this is why epoch 1 and epoch 2 durations are close to
each other rather than epoch 1 being disproportionately slower: the expensive
part of preprocessing is not repeated per-epoch, and the per-epoch lazy part is
cheap relative to GPU compute.
