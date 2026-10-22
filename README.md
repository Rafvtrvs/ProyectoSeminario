# ProyectoSeminario

Sistema asistido de visión artificial (Solemne 2 — CINF1P09).  
Grupo A: Raspberry Pi (inferencia ML) + placa (XIAO/Arduino) + cámara del prototipo.

## Estado actual (ML)

- Dataset unificado: `persona`, `poste_luz`, `senal_stop`, `senal_transito`
- Entrenamiento en Colab (YOLOv8n): notebook `notebooks/entrenamiento_baseline_v1.ipynb`
- Modelo elegido: **expB, imgsz=224** (`best.pt` en Google Drive, no en este repo)
- Comparación en test (mAP50): baseline 0.827 → A(160) 0.835 → **B(224) 0.873**

Los datos (`data/raw`, `data/processed`, `*.zip`) y los pesos (`*.pt`) **no** se suben a Git.

## Scripts

- `scripts/curate/unify_datasets.py` — unifica datasets YOLO → `data/processed/`
- `scripts/curate/check_dataset.py` — conteos y muestras con cajas

## Nota

`train.py` + CML actuales siguen siendo el ejemplo de California Housing; se reemplazarán cuando el pipeline de evaluación del detector esté listo.
