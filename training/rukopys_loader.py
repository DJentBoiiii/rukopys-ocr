"""Loader for the RUKOPYS dataset (train split for finetuning, test split for eval).

Reads the raw `train/metadata.jsonl` + `train/images/*.jpg` (and the `test/`
equivalent) that were fetched once via `download_rukopys()`, avoiding the
`datasets` Arrow-caching overhead for this already-small dataset (~1330
train + 385 test images).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

from preprocessing.deskew import deskew
from training.datasets_common import resize_to_fixed_height, to_grayscale_array

# archive scans can exceed Pillow's default decompression-bomb pixel limit;
# these come from our own trusted dataset download, so raise it instead of
# disabling the check entirely.
Image.MAX_IMAGE_PIXELS = 300_000_000

REPO_ID = "UkrainianCatholicUniversity/rukopys"
DEFAULT_CACHE_DIR = "data_cache/rukopys"
VAL_SPLIT_PATH = Path(__file__).resolve().parent / "val_split.json"
VAL_SPLIT_SIZE = 300
VAL_SPLIT_SEED = 42

# archive documents (CDAVO, 1919-1935) are the target domain: oversample them
SOURCE_WEIGHTS = {
    "archive": 4.0,
    "dictation": 1.0,
    "school": 1.0,
    "university": 1.0,
}


def download_rukopys(cache_dir: str = DEFAULT_CACHE_DIR) -> Path:
    """Download only the `train/` and `test/` prefixes (skips the much larger
    `silver/` split, which is not needed for this task).
    """
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["train/*", "test/*"],
        local_dir=cache_dir,
    )
    return Path(path)


def _load_split_regions(root: Path, split: str) -> List[dict]:
    """Return a flat list of {"image_path", "text", "source"} for every
    `handwritten` region in the given split ("train" or "test").
    """
    metadata_path = root / split / "metadata.jsonl"
    regions = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            image_path = root / split / record["file_name"]
            for region in record["regions"]:
                if region["type"] != "handwritten":
                    continue
                text = region["text"].strip()
                if not text:
                    continue
                regions.append({
                    "image_path": image_path,
                    "bbox": region["bbox"],
                    "text": text,
                    "source": record["source"],
                })
    return regions


def _crop_region(image_path: Path, bbox: List[int]) -> np.ndarray:
    with Image.open(image_path) as img:
        crop = img.crop(tuple(bbox))
        return to_grayscale_array(crop)


def _region_id(region: dict) -> str:
    """Stable identifier for a region: file name + bbox (unique within a split)."""
    bbox = ",".join(str(v) for v in region["bbox"])
    return f"{Path(region['image_path']).name}|{bbox}"


def build_or_load_val_ids(
    root: Path,
    n: int = VAL_SPLIT_SIZE,
    seed: int = VAL_SPLIT_SEED,
) -> List[str]:
    """Build a fixed validation split (region ids) from RUKOPYS train, stratified
    by `source` where possible, and persist it to `VAL_SPLIT_PATH` so the same
    validation subset is reused across runs instead of being resampled.
    """
    if VAL_SPLIT_PATH.exists():
        return json.loads(VAL_SPLIT_PATH.read_text(encoding="utf-8"))

    regions = _load_split_regions(root, "train")
    by_source: dict = {}
    for region in regions:
        by_source.setdefault(region["source"], []).append(region)

    rng = random.Random(seed)
    total = len(regions)
    remaining = n
    chosen: List[dict] = []
    sources = sorted(by_source.keys())
    for i, source in enumerate(sources):
        pool = by_source[source]
        if i == len(sources) - 1:
            take = remaining
        else:
            take = min(round(n * len(pool) / total), remaining, len(pool))
        sampled = rng.sample(pool, take) if take > 0 else []
        chosen.extend(sampled)
        remaining -= len(sampled)

    ids = [_region_id(r) for r in chosen]
    VAL_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VAL_SPLIT_PATH.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
    return ids


def load_rukopys_val_set(root: Path, apply_deskew: bool = True) -> List[dict]:
    """Load the fixed validation subset of RUKOPYS train (never used for
    training itself), pre-cropped and deskewed, for periodic val WER/CER checks.
    """
    val_ids = set(build_or_load_val_ids(root))
    regions = _load_split_regions(root, "train")
    val_regions = [r for r in regions if _region_id(r) in val_ids]

    samples = []
    for region in tqdm(val_regions, desc="loading val regions"):
        image = _crop_region(region["image_path"], region["bbox"])
        if apply_deskew and image.shape[0] > 8 and image.shape[1] > 8:
            try:
                image = deskew(image)
            except Exception:
                pass
        # resize to fixed height immediately -- holding full-resolution scans
        # for all regions simultaneously is what was OOM-killing the process
        # before it even finished loading (no swap on this machine).
        image = resize_to_fixed_height(image)
        samples.append({"image": image, "text": region["text"], "source": region["source"]})
    return samples


def load_rukopys_train_subset(
    root: Path,
    max_examples: int,
    seed: int = 42,
    apply_deskew: bool = True,
    exclude_ids: set | None = None,
) -> List[Tuple[np.ndarray, str]]:
    """Load a training subset from RUKOPYS train, sampled with a higher
    weight on `source: archive` examples (without fully excluding others).

    `exclude_ids` (region ids, see `_region_id`) are removed from the pool
    before sampling, e.g. to keep the fixed validation subset out of training.
    """
    regions = _load_split_regions(root, "train")
    if exclude_ids:
        regions = [r for r in regions if _region_id(r) not in exclude_ids]
    rng = random.Random(seed)

    weights = [SOURCE_WEIGHTS.get(r["source"], 1.0) for r in regions]
    n = min(max_examples, len(regions))
    chosen_idx = set()
    # weighted sampling without replacement
    pool = list(range(len(regions)))
    pool_weights = list(weights)
    while len(chosen_idx) < n and pool:
        idx = rng.choices(range(len(pool)), weights=pool_weights, k=1)[0]
        chosen_idx.add(pool[idx])
        del pool[idx]
        del pool_weights[idx]

    samples = []
    for i in tqdm(chosen_idx, desc="loading train examples"):
        region = regions[i]
        image = _crop_region(region["image_path"], region["bbox"])
        if apply_deskew and image.shape[0] > 8 and image.shape[1] > 8:
            try:
                image = deskew(image)
            except Exception:
                pass
        # resize to fixed height immediately -- see comment in
        # `load_rukopys_val_set` for why this can't wait until later.
        image = resize_to_fixed_height(image)
        samples.append((image, region["text"]))
    return samples


def load_rukopys_test_set(root: Path, apply_deskew: bool = True) -> List[dict]:
    """Load the full official RUKOPYS test split, for final evaluation only.

    Returns a list of dicts with image, text, and source, so callers can
    break results down by source (e.g. archive vs. other).
    """
    regions = _load_split_regions(root, "test")
    samples = []
    for region in tqdm(regions, desc="loading test regions"):
        image = _crop_region(region["image_path"], region["bbox"])
        if apply_deskew and image.shape[0] > 8 and image.shape[1] > 8:
            try:
                image = deskew(image)
            except Exception:
                pass
        image = resize_to_fixed_height(image)
        samples.append({"image": image, "text": region["text"], "source": region["source"]})
    return samples
