# ProyectoSeminario

**Solemne 2 — Programación de Microprocesadores (CINF1P09)**  
**Grupo A:** Sistema asistido de visión artificial

**Idea en una frase:** una cámara mira el entorno, un modelo de inteligencia artificial dice qué ve (persona, señal, etc.), y una placa ayuda a disparar la captura y avisar con LED o buzzer. El “cerebro” del modelo corre en la **Raspberry Pi**, no en el XIAO.

---

## ¿Qué hay en este repositorio?

| Carpeta / archivo | ¿Para qué sirve? |
|---|---|
| `notebooks/entrenamiento_baseline_v1.ipynb` | Notebook de Google Colab para **entrenar** el detector (YOLOv8). Ahí se hicieron los experimentos baseline, A y B. |
| `scripts/curate/unify_datasets.py` | Junta varios datasets de Roboflow en **uno solo** con 4 clases comunes. |
| `scripts/curate/check_dataset.py` | Revisa el dataset: cuenta cajas por clase y dibuja ejemplos para verificar etiquetas. |
| `configs/classes.yaml` | Lista de clases del proyecto y cómo se mapean desde cada dataset. |
| `data/README.md` | Explica cómo se organizan los datos (crudos → unificados). Las fotos en sí **no** van en Git (pesan mucho). |
| `reports/baseline_comparison.md` | Tabla con los resultados de los 3 entrenamientos y cuál modelo elegimos. |
| `train.py` | Script viejo de prueba (precios de casas). Sirvió para probar CML; **no** es el modelo de visión. |
| `.github/workflows/cml.yaml` | Automatización en GitHub que hoy corre ese `train.py` de prueba. Más adelante se cambiará al detector. |
| `requeriments.txt` | Librerías de Python del ejemplo antiguo (pandas, sklearn, matplotlib). |
| `.gitignore` | Evita subir ZIPs, fotos del dataset y archivos `.pt` (pesos del modelo). |

### Lo que está fuera de Git (pero sí lo usamos)

- **Fotos del dataset** → en tu PC (`data/`) o en el ZIP de Drive.
- **Modelo entrenado `best.pt`** → en Google Drive (`ProyectoSeminario/runs/expB_224_e150/`).

---

## Modelo que usaremos (por ahora)

- Nombre: **YOLOv8n, experimento B**
- Resolución: **224×224**
- Clases: `persona`, `poste_luz`, `senal_stop`, `senal_transito`
- En test: mAP50 **0.873** (mejor que baseline y que A)

Para la **demo en sala** conviene mostrar sobre todo **persona** y **señal stop** (impresa).

---

## Siguientes pasos (según la Solemne)

Orden recomendado para esta semana:

1. **Confirmar el hardware (por escrito)**  
   Raspberry Pi (modelo), cámara (USB / Camera Module / OV2640 del XIAO), placa (XIAO o Arduino), cables.

2. **Armar el flujo mínimo**  
   Placa dispara → Pi captura foto → modelo detecta → se guarda un registro en CSV → LED o buzzer responde.

3. **Comunicación real Pi ↔ placa**  
   Mensajes claros por serial/USB (comandos, respuestas, errores).

4. **Demostrar una falla controlada**  
   Por ejemplo desconectar el cable y que el sistema avise y se recupere.

5. **Sacar ~100 fotos reales con la cámara del prototipo**  
   Con variaciones de luz, distancia y ángulo. Etiquetar a mano qué había en cada foto. **No** usar la cámara de la laptop para la entrega.

6. **Calcular métricas sobre esas 100 fotos**  
   Matriz de confusión, precisión, recall y F1 (lo que pide el profesor).

7. **Ensayar la demo**  
   Incluyendo la falla controlada.

### Lo que ya está avanzado

- Datasets unidos y revisados  
- Modelo baseline entrenado y comparado (elegimos B @ 224)

### Lo que aún falta (lo más importante)

- Código en la Raspberry Pi  
- Firmware de la placa  
- CSV de eventos  
- Falla controlada  
- Las 100 observaciones reales  

---

## Cómo abrir el notebook de entrenamiento

1. Sube `processed_v1.zip` a Google Drive (si aún no está).  
2. Abre el `.ipynb` en Colab (o súbelo desde este repo).  
3. Activa GPU (T4).  
4. Monta Drive y sigue las celdas.

Los pesos finales no están en GitHub: descárgalos desde Drive (`best.pt` de la carpeta `expB_224_e150`).

---

## Integrantes

Mateo Cortés Medina, Rafael Rodríguez Tovar, Josefa Iturra González.
