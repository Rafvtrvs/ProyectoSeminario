"""
Unifica datasets YOLO en data/raw/ hacia data/processed/ con la taxonomía v1.

Uso (desde la raíz del repo):
    python scripts/curate/unify_datasets.py

No modifica data/raw/. Sobrescribe data/processed/ si ya existe contenido.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Clases finales (deben coincidir con configs/classes.yaml)
TARGET_NAMES = {
    0: "persona",
    1: "poste_luz",
    2: "senal_stop",
    3: "senal_transito",
}
NAME_TO_ID = {v: k for k, v in TARGET_NAMES.items()}

# Por cada fuente: nombre de clase original -> clase unificada (None = ignorar)
# Los índices YOLO de cada zip se resuelven leyendo su data.yaml
SOURCE_SPECS = {
    "pedestrian": {
        "folder": ROOT / "data" / "raw" / "pedestrian",
        "map": {"person": "persona"},
        "ignore": set(),
        # Este zip no trae valid/: se reserva una fracción del train
        "holdout_valid_from_train": 0.15,
    },
    "street_signs": {
        "folder": ROOT / "data" / "raw" / "street_signs",
        "map": {
            "stop": "senal_stop",
            "speedlimit": "senal_transito",
        },
        "ignore": {"crosswalk", "trafficlight"},
        "holdout_valid_from_train": 0.0,
    },
    "lamp_post": {
        "folder": ROOT / "data" / "raw" / "lamp_post",
        "map": {"lamp post": "poste_luz"},
        "ignore": set(),
        "holdout_valid_from_train": 0.0,
    },
}

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_yolo_names(data_yaml: Path) -> list[str]:
    """Lee la lista names: [...] del data.yaml de Roboflow sin depender de PyYAML."""
    text = data_yaml.read_text(encoding="utf-8")
    match = re.search(r"names:\s*\[([^\]]*)\]", text)
    if not match:
        raise ValueError(f"No se encontró names: [...] en {data_yaml}")
    inner = match.group(1)
    names = [n.strip().strip("'\"") for n in inner.split(",") if n.strip()]
    if not names:
        raise ValueError(f"Lista names vacía en {data_yaml}")
    return names


def build_index_remap(source_names: list[str], class_map: dict[str, str], ignore: set[str]) -> dict[int, int | None]:
    """
    old_class_id -> new_class_id | None (None = descartar esa caja).
    """
    remap: dict[int, int | None] = {}
    for old_id, name in enumerate(source_names):
        if name in ignore:
            remap[old_id] = None
            continue
        if name not in class_map:
            raise KeyError(
                f"Clase '{name}' no está en map ni en ignore. "
                f"map={class_map}, ignore={ignore}"
            )
        target_name = class_map[name]
        remap[old_id] = NAME_TO_ID[target_name]
    return remap


def find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        # Roboflow a veces usa mayúsculas en la extensión
        candidate = images_dir / f"{stem}{ext.upper()}"
        if candidate.exists():
            return candidate
    return None


def rewrite_label_lines(label_path: Path, remap: dict[int, int | None]) -> list[str]:
    """Devuelve líneas YOLO con IDs remapeados; omite cajas ignoradas."""
    out: list[str] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_id = int(parts[0])
        if old_id not in remap:
            continue
        new_id = remap[old_id]
        if new_id is None:
            continue
        parts[0] = str(new_id)
        out.append(" ".join(parts))
    return out


def collect_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    """Lista (imagen, etiqueta) de un split YOLO."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return []

    pairs: list[tuple[Path, Path]] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = find_image_for_label(images_dir, label_path.stem)
        if image_path is None:
            continue
        pairs.append((image_path, label_path))
    return pairs


def ensure_processed_dirs(out_root: Path) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in SPLITS:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_pair(
    image_path: Path,
    new_lines: list[str],
    out_root: Path,
    split: str,
    prefix: str,
    counters: dict[str, int],
) -> None:
    """Copia imagen + escribe etiqueta con nombre único prefijado por fuente."""
    counters[split] += 1
    stem = f"{prefix}_{counters[split]:06d}"
    dest_img = out_root / split / "images" / f"{stem}{image_path.suffix.lower()}"
    dest_lbl = out_root / split / "labels" / f"{stem}.txt"
    shutil.copy2(image_path, dest_img)
    dest_lbl.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")


def process_source(
    name: str,
    spec: dict,
    out_root: Path,
    counters: dict[str, int],
    stats: dict,
    seed: int,
) -> None:
    folder: Path = spec["folder"]
    data_yaml = folder / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Falta {data_yaml}")

    source_names = parse_yolo_names(data_yaml)
    remap = build_index_remap(source_names, spec["map"], spec["ignore"])
    print(f"\n[{name}] clases origen: {source_names}")
    print(f"[{name}] remap índices: {remap}")

    stats[name] = {s: {"images": 0, "boxes": 0, "skipped_empty": 0} for s in SPLITS}

    # train (+ holdout a valid si aplica)
    train_pairs = collect_pairs(folder / "train")
    rng = random.Random(seed + hash(name) % 10_000)
    holdout = float(spec.get("holdout_valid_from_train", 0.0))

    if holdout > 0 and train_pairs:
        rng.shuffle(train_pairs)
        n_valid = max(1, int(len(train_pairs) * holdout))
        valid_holdout = train_pairs[:n_valid]
        train_pairs = train_pairs[n_valid:]
        print(f"[{name}] holdout valid desde train: {n_valid} imágenes")
    else:
        valid_holdout = []

    split_jobs: list[tuple[str, list[tuple[Path, Path]]]] = [
        ("train", train_pairs),
        ("valid", valid_holdout + collect_pairs(folder / "valid")),
        ("test", collect_pairs(folder / "test")),
    ]

    for split, pairs in split_jobs:
        for image_path, label_path in pairs:
            new_lines = rewrite_label_lines(label_path, remap)
            if not new_lines:
                # Imagen que solo tenía clases ignoradas (ej. solo crosswalk)
                stats[name][split]["skipped_empty"] += 1
                continue
            copy_pair(image_path, new_lines, out_root, split, name, counters)
            stats[name][split]["images"] += 1
            stats[name][split]["boxes"] += len(new_lines)


def write_data_yaml(out_root: Path) -> None:
    names_list = [TARGET_NAMES[i] for i in range(len(TARGET_NAMES))]
    content = (
        f"path: {out_root.as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"\n"
        f"nc: {len(names_list)}\n"
        f"names: {names_list}\n"
    )
    (out_root / "data.yaml").write_text(content, encoding="utf-8")


def print_stats(stats: dict, counters: dict[str, int]) -> None:
    print("\n=== Resumen ===")
    for source, splits in stats.items():
        print(f"\n{source}:")
        for split, s in splits.items():
            print(
                f"  {split:5s}  imágenes={s['images']:5d}  "
                f"cajas={s['boxes']:6d}  "
                f"omitidas_sin_clase_v1={s['skipped_empty']:5d}"
            )
    print("\nTotales en processed/:")
    for split in SPLITS:
        print(f"  {split:5s}  {counters[split]} imágenes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unifica datasets YOLO raw -> processed")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Carpeta de salida (default: data/processed)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla del holdout valid")
    args = parser.parse_args()

    out_root = args.out if args.out.is_absolute() else ROOT / args.out
    ensure_processed_dirs(out_root)

    counters = {s: 0 for s in SPLITS}
    stats: dict = {}

    for name, spec in SOURCE_SPECS.items():
        process_source(name, spec, out_root, counters, stats, args.seed)

    write_data_yaml(out_root)
    print_stats(stats, counters)
    print(f"\nListo. Dataset unificado en: {out_root}")
    print(f"Config YOLO: {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
