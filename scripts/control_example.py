"""Visualize a handful of RUKOPYS test examples: crop -> deskew -> OCR output
(raw and dictionary-corrected) vs. ground truth, saved as PNGs + a markdown
summary for direct inclusion in a report.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training import config
from training.evaluate import build_reconstruction_module, load_model_for_eval, predict
from training.rukopys_loader import _crop_region, _load_split_regions, download_rukopys
from preprocessing.deskew import deskew

OUTPUT_DIR = config.PROJECT_ROOT / "experiments" / "control_example"


def pick_regions(regions: list[dict]) -> list[dict]:
    """Pick 4 control regions (lightweight metadata, not yet cropped): one
    `archive`, one `dictation`, and two arbitrary ones (shortest and longest
    ground-truth text among the rest).
    """
    archive = next(r for r in regions if r["source"] == "archive")
    dictation = next(r for r in regions if r["source"] == "dictation")

    remaining = [r for r in regions if r is not archive and r is not dictation]
    shortest = min(remaining, key=lambda r: len(r["text"]))
    longest = max(remaining, key=lambda r: len(r["text"]))

    return [archive, dictation, shortest, longest]


def load_sample(region: dict) -> dict:
    image = _crop_region(region["image_path"], region["bbox"])
    if image.shape[0] > 8 and image.shape[1] > 8:
        try:
            image = deskew(image)
        except Exception:
            pass
    return {"image": image, "text": region["text"], "source": region["source"]}


def render_example(index: int, sample: dict, raw_text: str, corrected_text: str) -> Path:
    fig, (ax_img, ax_text) = plt.subplots(
        2, 1, figsize=(10, 4), gridspec_kw={"height_ratios": [2, 1.4]},
    )

    ax_img.imshow(sample["image"], cmap="gray")
    ax_img.set_title(f"example {index} (source: {sample['source']})")
    ax_img.axis("off")

    ax_text.axis("off")
    lines = [
        f"Ground truth: {sample['text']}",
        f"OCRProcessor (сирий вивід): {raw_text}",
        f"+ пост-корекція: {corrected_text}",
    ]
    ax_text.text(
        0.0, 1.0, "\n\n".join(lines),
        transform=ax_text.transAxes, va="top", ha="left",
        fontsize=11, fontfamily="DejaVu Sans", wrap=True,
    )

    out_path = OUTPUT_DIR / f"example_{index}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    download_rukopys(str(config.DATA_CACHE_DIR))
    regions = _load_split_regions(config.DATA_CACHE_DIR, "test")

    model, tokenizer, decoder = load_model_for_eval()
    reconstructor = build_reconstruction_module()

    chosen_regions = pick_regions(regions)

    summary_lines = ["# Контрольні приклади\n"]
    for i, region in enumerate(chosen_regions, start=1):
        sample = load_sample(region)
        raw_text = predict(model, decoder, sample["image"])
        corrected_text = reconstructor.reconstruct(raw_text)

        png_path = render_example(i, sample, raw_text, corrected_text)
        print(f"saved {png_path}")

        summary_lines.append(f"## Приклад {i} (source: {sample['source']})\n")
        summary_lines.append(f"- **Ground truth**: {sample['text']}")
        summary_lines.append(f"- **OCRProcessor (сирий вивід)**: {raw_text}")
        summary_lines.append(f"- **+ пост-корекція**: {corrected_text}\n")

    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
