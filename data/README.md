# Datos del proyecto

## Estructura

```text
data/
  raw/           # Descargas originales (sin modificar)
    pedestrian/
    street_signs/
    lamp_post/
    _extra/      # Datasets opcionales / no usados en v1
  interim/       # Unificación de clases (paso intermedio)
  processed/     # Dataset final listo para entrenar (train/val/test)
```

## Fuentes v1 (Roboflow, formato YOLO)

| Carpeta | Dataset | Clases originales |
|---|---|---|
| `raw/pedestrian` | Pedestrian Detector (Leonid) | peatones |
| `raw/street_signs` | Street-signs (Aiman Younis) | stop, crosswalk, trafficlight, speedlimit |
| `raw/lamp_post` | Lamp post Datasets (bgyz) | lamp post |

## Clases unificadas (objetivo)

- `persona`
- `poste_luz`
- `senal_stop`
- `senal_transito`

Los datos en `raw/` **no** van a Git (ver `.gitignore`).
