# Comparación de baselines (conjunto de test)

Dataset: `data/processed` (998 imágenes de test). Modelo: YOLOv8n.

| Modelo | imgsz | epochs | mAP50 | Recall | persona R | poste_luz R |
|---|---|---|---|---|---|---|
| baseline_v1_160 | 160 | 50 | 0.827 | 0.760 | 0.765 | 0.432 |
| expA_160_e150 | 160 | 150 | 0.835 | 0.796 | 0.788 | 0.481 |
| **expB_224_e150** | **224** | 150 | **0.873** | **0.833** | **0.841** | **0.554** |

**Decisión:** desplegar `expB_224_e150/weights/best.pt` en la Raspberry Pi con `imgsz=224`.

Pesos guardados en Google Drive (`ProyectoSeminario/runs/`), no en este repositorio.
