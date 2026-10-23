# Comparación de baselines (conjunto de test)

Modelo: YOLOv8n. Métricas en split `test`.

## v1 — dataset original (~998 imgs test)

| Modelo | imgsz | epochs | mAP50 | Recall | persona R | poste_luz R |
|---|---|---|---|---|---|---|
| baseline_v1_160 | 160 | 50 | 0.827 | 0.760 | 0.765 | 0.432 |
| expA_160_e150 | 160 | 150 | 0.835 | 0.796 | 0.788 | 0.481 |
| expB_224_e150 | 224 | 150 | 0.873 | 0.833 | 0.841 | 0.554 |

Pocas cajas de `senal_stop` en test (~13). Decisión inicial: expB @ 224.

## v2 — dataset con más stop (1143 imgs test)

Fuentes nuevas unificadas: `stop`, `stop_sing`, `stop_signal`, `traffic_stop` (+ v1).  
Run: `expB_224_e150-4` (Drive: `ProyectoSeminario/runs/expB_224_e150-4`).

| Clase | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.887 | 0.827 | **0.868** | 0.648 |
| persona | 0.858 | 0.828 | 0.891 | 0.597 |
| poste_luz | 0.752 | 0.543 | 0.614 | 0.333 |
| senal_stop | 0.956 | 0.981 | **0.991** | 0.893 |
| senal_transito | 0.982 | 0.957 | 0.976 | 0.769 |

El mAP50 global (0.868) es comparable al expB v1 (0.873) sobre un test distinto y más grande.  
`senal_stop` pasa a ser la clase más fuerte (mAP50 0.991; 158 imgs / 162 cajas en test).  
`poste_luz` sigue siendo la clase más débil.

**Decisión actual:** desplegar `expB_224_e150-4/weights/best.pt` en la xiao bajo un tinyML con Edge Impulse`.

Pesos en Google Drive (`ProyectoSeminario/runs/`), no en este repositorio.
