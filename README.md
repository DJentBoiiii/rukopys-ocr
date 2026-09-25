# rukopys-ocr

Модель розпізнавання рукописного тексту (`OCRProcessor`) для реконструкції
українських архівних рукописів. Ізольована підзадача: препроцесинг
зображення рядка тексту + CRNN (CNN + BiLSTM + CTC) модель, навчена на CPU
за один нічний прогін.

Статус даних, епох і результатів — див. [experiments/results.md](experiments/results.md)
(заповнюється після тренування).

## Структура

```
preprocessing/   # бінаризація (Sauvola), корекція нахилу (Hough)
model/            # CRNN модель, CTC-декодер, символьний токенізатор
training/         # завантажувачі датасетів, тренувальний цикл, evaluate.py
experiments/      # results.md з фінальними метриками
tests/            # pytest юніт-тести
checkpoints/      # ваги моделі (НЕ в git, див. .gitignore)
```

## Залежності

Python 3.11+ (перевірено на 3.14), PyTorch (CPU), `datasets`, OpenCV,
scikit-image, jiwer, pytesseract.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install datasets opencv-python-headless scikit-image jiwer pytesseract pillow numpy pytest tqdm huggingface_hub
```

Для порівняння з Tesseract потрібен системний пакет:

```bash
# Arch / EndeavourOS
sudo pacman -S tesseract tesseract-data-ukr

# Debian / Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-ukr
```

Якщо `tesseract` недоступний у середовищі, `training/evaluate.py`
пропускає цю частину порівняння з явним попередженням у лозі, решта
оцінки (власна модель) виконується як завжди.

## Запуск

```bash
# 1. попереднє навчання на synthetic-cyrillic-large (streaming)
python -m training.train --stage pretrain

# 2. донавчання на RUKOPYS train
python -m training.train --stage finetune

# 3. оцінка на офіційному RUKOPYS test (WER/CER, своя модель + tesseract)
python -m training.evaluate

# тести
pytest -q
```

Ваги моделі зберігаються в `checkpoints/` локально і не потрапляють у git.

### Нічний прогін з ОС-лімітом

Весь пайплайн (pretrain -> finetune -> evaluate) можна виконати одним
викликом під жорстким лімітом часу на рівні ОС, щоб гарантовано не
перевищити доступне нічне вікно, навіть якщо якийсь етап зависне:

```bash
timeout 5h bash scripts/run_overnight.sh
```

Кожен етап регулярно зберігає checkpoint (див. `CHECKPOINT_EVERY_STEPS`
у `training/config.py`), тому навіть при спрацюванні `timeout` останній
збережений checkpoint можна використати для оцінки чи донавчання.


## Дані

Точні обсяги використаних даних, кількість епох і час тренування —
зафіксовані в [experiments/results.md](experiments/results.md).
