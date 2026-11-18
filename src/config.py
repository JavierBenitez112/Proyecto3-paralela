"""Constantes globales del proyecto. Unica fuente de rutas y parametros.

Mismos parametros que el Laboratorio 3 (ASL) para que los tiempos sean comparables.
La ruta del dataset crudo se puede cambiar con la variable de entorno ASL_DATA
(por ejemplo en Colab), sin tocar el codigo.
"""

import json
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# Carpeta que contiene asl_alphabet_train/<clase>/*.jpg
RUTA_TRAIN_CRUDO = Path(os.environ.get(
    "ASL_DATA",
    RAIZ.parent / "Laboratorio3_Deep_Learning" / "data" / "raw" / "asl_alphabet_train",
))

DIR_PROCESSED = RAIZ / "data" / "processed"
RUTA_TENSORES = DIR_PROCESSED / "tensores.npz"
# P3_RESULTADOS permite medir otro equipo sin pisar resultados/ (p. ej. resultados/rtx5070)
DIR_RESULTADOS = Path(os.environ.get("P3_RESULTADOS", RAIZ / "resultados"))
DIR_FIGURAS = DIR_RESULTADOS / "figuras"

SEMILLA = 123

CLASES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]
NUM_CLASES = len(CLASES)

IMGS_POR_CLASE = 600
RESOLUCION = 64
CANALES = 1
FORMA_ENTRADA = (RESOLUCION, RESOLUCION, CANALES)

PROP_TRAIN = 0.70
PROP_VAL = 0.15

# Medicion: mismas condiciones para la version secuencial y la paralela
REPETICIONES = 5
EPOCAS_BENCHMARK = 5
BATCH_SIZE = 64


def guardar_tiempos(fase, variante, tiempos_seg, **extra):
    """Escribe resultados/<fase>_<variante>.json. Contrato unico que lee 04_comparacion.py.

    fase     : "preparacion" | "entrenamiento" | "convolucion"
    variante : "secuencial" | "paralelo"
    extra    : cualquier dato adicional (n_imagenes, dispositivo, transferencia_seg, ...)
    """
    if not tiempos_seg:
        raise ValueError(f"{fase}/{variante}: no hay tiempos que guardar (repeticiones >= 1)")
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    registro = {
        "fase": fase,
        "variante": variante,
        "tiempos_seg": [round(float(t), 4) for t in tiempos_seg],
        "promedio_seg": round(sum(tiempos_seg) / len(tiempos_seg), 4),
        **extra,
    }
    ruta = DIR_RESULTADOS / f"{fase}_{variante}.json"
    ruta.write_text(json.dumps(registro, indent=2), encoding="utf-8")
    print(f"[guardado] {ruta.name} -> promedio {registro['promedio_seg']} s")
    return registro
