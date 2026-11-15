"""
Fase 1 del proyecto: PREPARACION DE DATOS (CPU, secuencial vs paralelo)
------------------------------------------------------------
Compara el preprocesamiento de imagenes del Laboratorio 3 contra una version con
multiprocessing. Es una fase estrictamente CPU: no usa GPU ni tensorflow.

  Lee : data/raw/asl_alphabet_train/<clase>/*.jpg  (ruta en cfg.RUTA_TRAIN_CRUDO)
  Hace: - arma el inventario en memoria (submuestra por clase y split train/val/test)
        - procesa cada imagen: grises, resize 64x64 LANCZOS, normaliza a [0, 1]
        - mide la version secuencial (for simple) y la paralela (Pool.map)
        - verifica que ambas dan exactamente el mismo resultado
  Escribe: data/processed/tensores.npz
           resultados/preparacion_secuencial.json y resultados/preparacion_paralelo.json

Ejecutar:  python src/01_preparacion.py [--imgs-por-clase N] [--repeticiones N]
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

import config as cfg

EXTENSION = "*.jpg"
FILTRO = Image.Resampling.LANCZOS
PARTICIONES = ["train", "val", "test"]


def construir_inventario(imgs_por_clase):
    """Devuelve {particion: (rutas, etiquetas)} igual que 01_particiones.py del lab."""
    if not cfg.RUTA_TRAIN_CRUDO.is_dir():
        sys.exit(f"[FALLO] no existe el dataset en {cfg.RUTA_TRAIN_CRUDO} (define ASL_DATA)")

    rng = np.random.default_rng(cfg.SEMILLA)
    n_train = int(imgs_por_clase * cfg.PROP_TRAIN)
    n_val = int(imgs_por_clase * cfg.PROP_VAL)
    inv = {p: ([], []) for p in PARTICIONES}

    for i, clase in enumerate(cfg.CLASES):
        rutas = sorted((cfg.RUTA_TRAIN_CRUDO / clase).glob(EXTENSION))
        if len(rutas) < imgs_por_clase:
            sys.exit(f"[FALLO] clase {clase}: {len(rutas)} imagenes, se piden {imgs_por_clase}")
        elegidas = rng.choice(len(rutas), size=imgs_por_clase, replace=False)
        rng.shuffle(elegidas)

        for orden, idx in enumerate(elegidas):
            p = "train" if orden < n_train else "val" if orden < n_train + n_val else "test"
            inv[p][0].append(str(rutas[idx]))
            inv[p][1].append(i)
    return inv


def procesar_imagen(ruta):
    """Abre un jpg y devuelve su matriz en grises, reescalada y normalizada."""
    with Image.open(ruta) as img:
        img = img.convert("L").resize((cfg.RESOLUCION, cfg.RESOLUCION), FILTRO)
        return np.asarray(img, dtype=np.float32) / 255.0


def secuencial(rutas):
    return [procesar_imagen(r) for r in rutas]


def paralelo(rutas):
    n = os.cpu_count()
    with Pool(n) as pool:
        return pool.map(procesar_imagen, rutas, chunksize=max(1, len(rutas) // (n * 4)))


def medir(funcion, rutas, repeticiones, nombre):
    """Ejecuta la funcion N veces; devuelve (tiempos, ultimo resultado)."""
    tiempos = []
    for r in range(repeticiones):
        t0 = time.perf_counter()
        res = funcion(rutas)
        tiempos.append(time.perf_counter() - t0)
        print(f"  {nombre} corrida {r + 1}/{repeticiones}: {tiempos[-1]:.2f} s")
    return tiempos, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs-por-clase", type=int, default=cfg.IMGS_POR_CLASE)
    ap.add_argument("--repeticiones", type=int, default=cfg.REPETICIONES)
    args = ap.parse_args()

    inv = construir_inventario(args.imgs_por_clase)
    # Se procesan todas las particiones juntas: el tiempo cubre todas las imagenes
    rutas = [r for p in PARTICIONES for r in inv[p][0]]
    n = len(rutas)
    print(f"[info] {n} imagenes, {os.cpu_count()} nucleos")

    # Pasada sin medir: deja los jpg en la cache de disco del sistema operativo. Sin esto
    # la primera corrida secuencial paga la lectura fria y el speedup sale inflado.
    print("[info] pasada de calentamiento (no se mide)")
    paralelo(rutas)

    t_sec, x_sec = medir(secuencial, rutas, args.repeticiones, "secuencial")
    t_par, x_par = medir(paralelo, rutas, args.repeticiones, "paralelo  ")

    assert np.array_equal(np.stack(x_sec), np.stack(x_par)), "paralelo != secuencial"
    print("[ok] la salida paralela es identica a la secuencial")

    # Reparte el resultado (ya ordenado train, val, test) en tensores
    tensores, ini = {}, 0
    for p in PARTICIONES:
        k = len(inv[p][0])
        tensores[f"x_{p}"] = np.stack(x_par[ini:ini + k])[..., np.newaxis].astype(np.float32)
        tensores[f"y_{p}"] = np.array(inv[p][1], dtype=np.int16)
        ini += k

    cfg.RUTA_TENSORES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.RUTA_TENSORES, clases=np.array(cfg.CLASES), **tensores)
    print(f"[guardado] {cfg.RUTA_TENSORES.name} -> {cfg.RUTA_TENSORES.stat().st_size / 1024**2:.0f} MB")

    cfg.guardar_tiempos("preparacion", "secuencial", t_sec, n_imagenes=n, nucleos=1)
    cfg.guardar_tiempos("preparacion", "paralelo", t_par, n_imagenes=n, nucleos=os.cpu_count())
    print(f"[speedup] {np.mean(t_sec) / np.mean(t_par):.2f}x")


if __name__ == "__main__":
    main()
