"""
Verifica el dataset unificado en data/processed/ antes de entrenar.

Uso (desde la raíz del repo):
    python scripts/curate/check_dataset.py

Genera:
    - Resumen por clase y split en consola
    - reports/dataset_check/summary.txt
    - reports/dataset_check/samples/<clase>/*.jpg con las cajas dibujadas
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Resolución de entrada prevista para el modelo del XIAO
TARGET_INPUT_PX = 160
# Una caja menor a esto (en px a TARGET_INPUT_PX) es difícil de detectar
TINY_BOX_PX = 8

COLORS = [(255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 200, 0)]


def parse_names(data_yaml: Path) -> list[str]:
    text = data_yaml.read_text(encoding="utf-8")
    match = re.search(r"names:\s*\[([^\]]*)\]", text)
    if not match:
        raise ValueError(f"No se encontró names: [...] en {data_yaml}")
    return [n.strip().strip("'\"") for n in match.group(1).split(",") if n.strip()]


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_boxes(label_path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    boxes = []
    problems = []
    for i, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            problems.append(f"{label_path.name}:{i} tiene {len(parts)} valores (se esperan 5)")
            continue
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            problems.append(f"{label_path.name}:{i} coordenadas fuera de [0,1]")
        boxes.append((cls, x, y, w, h))
    return boxes, problems


def draw_sample(image_path: Path, boxes, names: list[str], out_path: Path) -> None:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        for cls, x, y, w, h in boxes:
            x0, y0 = (x - w / 2) * width, (y - h / 2) * height
            x1, y1 = (x + w / 2) * width, (y + h / 2) * height
            color = COLORS[cls % len(COLORS)]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            draw.text((x0 + 3, max(0, y0 - 12)), names[cls], fill=color)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica data/processed antes de entrenar")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = args.data if args.data.is_absolute() else ROOT / args.data
    names = parse_names(data_root / "data.yaml")
    out_dir = ROOT / "reports" / "dataset_check"

    boxes_per_class = {s: defaultdict(int) for s in SPLITS}
    images_per_class = {s: defaultdict(int) for s in SPLITS}
    tiny_per_class = defaultdict(int)
    images_per_split = {s: 0 for s in SPLITS}
    examples_by_class = defaultdict(list)
    problems: list[str] = []
    missing_images = 0

    for split in SPLITS:
        images_dir = data_root / split / "images"
        for label_path in sorted((data_root / split / "labels").glob("*.txt")):
            image_path = find_image(images_dir, label_path.stem)
            if image_path is None:
                missing_images += 1
                continue
            boxes, label_problems = read_boxes(label_path)
            problems.extend(label_problems)
            images_per_split[split] += 1

            for cls in {b[0] for b in boxes}:
                images_per_class[split][cls] += 1
                examples_by_class[cls].append((image_path, boxes))

            for cls, _, _, w, h in boxes:
                if cls >= len(names):
                    problems.append(f"{label_path.name}: clase {cls} no existe en data.yaml")
                    continue
                boxes_per_class[split][cls] += 1
                if split == "train" and min(w, h) * TARGET_INPUT_PX < TINY_BOX_PX:
                    tiny_per_class[cls] += 1

    lines = []
    lines.append("=== Imágenes por split ===")
    for split in SPLITS:
        lines.append(f"  {split:5s}  {images_per_split[split]}")

    lines.append("\n=== Cajas por clase (imágenes con esa clase entre paréntesis) ===")
    header = f"  {'clase':16s}" + "".join(f"{s:>18s}" for s in SPLITS)
    lines.append(header)
    for cls, name in enumerate(names):
        row = f"  {name:16s}"
        for split in SPLITS:
            row += f"{boxes_per_class[split][cls]:>10d} ({images_per_class[split][cls]:>5d})"
        lines.append(row)

    lines.append(
        f"\n=== Cajas diminutas en train (lado < {TINY_BOX_PX}px si la entrada es {TARGET_INPUT_PX}px) ==="
    )
    for cls, name in enumerate(names):
        total = boxes_per_class["train"][cls]
        tiny = tiny_per_class[cls]
        pct = 100 * tiny / total if total else 0
        lines.append(f"  {name:16s} {tiny:6d} de {total:6d}  ({pct:5.1f} %)")

    lines.append("\n=== Problemas ===")
    lines.append(f"  Etiquetas sin imagen: {missing_images}")
    lines.append(f"  Líneas con errores:   {len(problems)}")
    lines.extend(f"    - {p}" for p in problems[:20])
    if len(problems) > 20:
        lines.append(f"    ... y {len(problems) - 20} más")

    summary = "\n".join(lines)
    print(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")

    rng = random.Random(args.seed)
    for cls, name in enumerate(names):
        examples = examples_by_class[cls]
        picked = rng.sample(examples, min(args.samples_per_class, len(examples)))
        for i, (image_path, boxes) in enumerate(picked, start=1):
            draw_sample(image_path, boxes, names, out_dir / "samples" / name / f"{i:02d}_{image_path.name}")

    print(f"\nMuestras con cajas dibujadas en: {out_dir / 'samples'}")


if __name__ == "__main__":
    main()
