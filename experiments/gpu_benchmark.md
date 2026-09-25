
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

## Крок 4: розрахунок бюджету даних/епох (2026-09-25, epoch ~1790367689)

Виміряні величини (з бенчмарку вище, `epoch 2` як стійка compute-only оцінка):
- one-time preprocessing (crop+deskew): `prep_per_example = 0.0424 s/example`
- steady-state compute (forward+backward+optimizer, batch=16, GPU): `compute_per_example = 0.0028 s/example`
- per-epoch overhead (val WER/CER на 300 прикладах + checkpoint saves): виміряно окремо,
  `val_eval ≈ 1.86s/epoch`, `checkpoint_save ≈ 0.015s × ~8 збережень/епоху ≈ 0.12s/epoch`
  → округлено до `overhead_per_epoch ≈ 2s`

Реальний пул даних (виміряно напряму, а не з інструкції): `_load_split_regions(train)`
дає **21527** непорожніх handwritten-регіонів; за вирахуванням фіксованого
валідаційного спліту (300) залишається **21227** прикладів для тренування
(інструкція згадувала ~25651, але фактичний підрахунок по кешованих даних
дає 21527 — довіряємо виміряному числу).

Бюджет часу (від старту завдання, epoch=1790366647, дедлайн = start+4h = 1790381047):
```
remaining_now      = deadline - now_at_step4        = 1790381047 - 1790367689 = 13358 s
reserve_for_step4   = 120 s   (комміт/пуш конфігу)
remaining_after_4   = 13358 - 120                    = 13238 s
usable_budget       = remaining_after_4 * 0.85 (15% запас) = 11252 s
```

Формула: `T(N, E) = prep_per_example * N + E * (compute_per_example * N + overhead_per_epoch)`

Для повного пулу `N = 21227`:
```
prep(N)      = 21227 * 0.0424        = 900 s   (15.0 хв, один раз)
per_epoch(N) = 21227 * 0.0028 + 2    = 61.4 s
E_max        = (usable_budget - prep(N)) / per_epoch(N)
             = (11252 - 900) / 61.4
             ≈ 168 epochs (теоретичний максимум у бюджет)
```

Обрано **E = 150** (з запасом нижче теоретичного максимуму 168, додатковий
буфер ≈ 1142 s / 19 хв понад вже закладені 15%):
```
T(21227, 150) = 900 + 150 * 61.4 = 10110 s ≈ 168.5 хв ≈ 2.81 год
```

Це вкладається в час, що лишився до 4-годинної позначки, із запасом.
Прискорення GPU (~33.5x за Кроком 3) значно вище порогу зупинки (3x), тому
форсування повного прогону виправдане.

**Підсумок конфігурації (training/config.py):**
- `FINETUNE_MAX_EXAMPLES = 21227` (весь доступний пул train-регіонів мінус 300 валідаційних)
- `FINETUNE_EPOCHS = 150`
- `FINETUNE_CHECKPOINT_GPU = checkpoints/finetune_gpu.pt` (окремий шлях,
  не чіпає ні `finetune.pt`, ні `finetune_v2.pt`)

Застереження: 150 епох — значно більше за попередній CPU-конфіг (20 епох на
18000 прикладах) виключно тому, що GPU-обчислення на порядки дешевші за
препроцесинг; ризик перенавчання без аугментації даних теоретично існує,
але попередній CPU-прогін (WER 87.7% після лише 10 епох на 6000 прикладів)
радше вказує на недонавчену модель, тож додаткові епохи практичніше
допоможуть, ніж нашкодять. Чекпоінти зберігаються кожні `CHECKPOINT_EVERY_STEPS`
кроків і після кожної епохи, тож проміжні стани доступні для відкату за
потреби.

## Крок 5: запуск

```
now_at_launch = 1790367763
deadline      = 1790381047
timeout_sec   = deadline - now_at_launch = 13284 s (3.69 год) -- жорсткий stop-cap для `timeout`
```
Очікувана тривалість самого прогону (`T(21227, 150) ≈ 10110 s`) значно менша за
`timeout_sec`, тобто `timeout` тут — лише запобіжник на випадок, якщо реальна
швидкість виявиться гіршою за виміряну на короткому бенчмарку; природне
завершення очікується раніше дедлайну.


