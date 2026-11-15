"""
Fase "entrenamiento": CNN del laboratorio, CPU (secuencial) vs GPU CUDA (paralelo).

CUDA se usa estrictamente para la matematica de la red (convoluciones, densas, Adam)
a traves de TensorFlow/cuDNN. Ambas variantes hacen exactamente el mismo trabajo:
mismo modelo (cnn_base del laboratorio), misma semilla, mismas epocas fijas y sin
EarlyStopping, para que la unica diferencia sea el dispositivo.

Lee    : cfg.RUTA_TENSORES (o --tensores PATH), generado por 01_preparacion.py
Hace   : imprime el hardware detectado, entrena en CPU y, si hay GPU, tambien en GPU
Escribe: resultados/entrenamiento_secuencial.json y resultados/entrenamiento_paralelo.json
Ejecutar: python src/02_entrenamiento.py [--repeticiones 3] [--epocas 5] [--tensores PATH]
"""

import argparse
import shutil
import subprocess
import time

import numpy as np
import tensorflow as tf

import config as cfg


def construir_modelo():
    """cnn_base del laboratorio, igual capa por capa."""
    capas = [tf.keras.layers.Input(cfg.FORMA_ENTRADA)]
    for filtros in (32, 64, 64):
        capas.append(tf.keras.layers.Conv2D(filtros, 3, activation="relu", padding="same"))
        capas.append(tf.keras.layers.MaxPooling2D(2))
    capas += [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(cfg.NUM_CLASES, activation="softmax"),
    ]
    modelo = tf.keras.Sequential(capas)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def entrenar_en(dispositivo, datos, epocas):
    """Entrena en "/CPU:0" o "/GPU:0". Devuelve (segundos de fit, val_accuracy final)."""
    tf.keras.utils.set_random_seed(cfg.SEMILLA)
    # tf.device("/CPU:0") basta para forzar CPU aunque haya GPU presente
    with tf.device(dispositivo):
        modelo = construir_modelo()
        if "GPU" in dispositivo:
            # calentamiento sin medir: inicializacion de CUDA/cuDNN y compilacion de kernels
            modelo.fit(datos["x_train"], datos["y_train"], epochs=1,
                       batch_size=cfg.BATCH_SIZE, verbose=0)
            # el calentamiento ya entreno 1 epoca: se reinicia el modelo con la misma semilla
            tf.keras.utils.set_random_seed(cfg.SEMILLA)
            modelo = construir_modelo()
        inicio = time.perf_counter()
        historial = modelo.fit(
            datos["x_train"], datos["y_train"],
            validation_data=(datos["x_val"], datos["y_val"]),
            epochs=epocas,
            batch_size=cfg.BATCH_SIZE,
            verbose=2,
        )
        segundos = time.perf_counter() - inicio
    return segundos, float(historial.history["val_accuracy"][-1])


def medir(dispositivo, datos, epocas, repeticiones):
    tiempos, accs = [], []
    for i in range(repeticiones):
        seg, acc = entrenar_en(dispositivo, datos, epocas)
        print(f"[info] {dispositivo} repeticion {i + 1}/{repeticiones}: {seg:.1f} s, val_acc {acc:.4f}")
        tiempos.append(seg)
        accs.append(acc)
    return tiempos, accs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeticiones", type=int, default=3)
    ap.add_argument("--epocas", type=int, default=cfg.EPOCAS_BENCHMARK)
    ap.add_argument("--tensores", default=cfg.RUTA_TENSORES)
    args = ap.parse_args()

    # evidencia de hardware reconocido por el entorno
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        # sin esto TensorFlow reserva toda la VRAM de golpe (4 GB en una RTX 3050 Ti)
        # y nvidia-smi no muestra el uso real de memoria
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[hardware] GPUs fisicas: {gpus}")
    print(f"[hardware] tf.test.gpu_device_name(): '{tf.test.gpu_device_name()}'")

    datos = np.load(args.tensores, allow_pickle=False)
    print(f"[cargado]  train {len(datos['x_train'])}, val {len(datos['x_val'])}")

    print("[info] variante secuencial (CPU)")
    t_cpu, acc_cpu = medir("/CPU:0", datos, args.epocas, args.repeticiones)
    cfg.guardar_tiempos("entrenamiento", "secuencial", t_cpu,
                        dispositivo="CPU", epocas=args.epocas, val_accuracy=round(acc_cpu, 4))

    if not gpus:
        print("[aviso] no hay GPU CUDA disponible, se omite la variante paralela")
        return

    nombre = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
    print(f"[info] variante paralela (GPU: {nombre})")
    # evidencia: registra uso y memoria de la GPU cada segundo mientras entrena
    ruta_log = cfg.DIR_RESULTADOS / "nvidia_smi_log.csv"
    monitor = None
    if shutil.which("nvidia-smi"):
        monitor = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total",
             "--format=csv", "-l", "1"], stdout=open(ruta_log, "w"))
    try:
        t_gpu, acc_gpu = medir("/GPU:0", datos, args.epocas, args.repeticiones)
    finally:
        if monitor:
            monitor.terminate()
            print(f"[guardado] {ruta_log.name}")
    cfg.guardar_tiempos("entrenamiento", "paralelo", t_gpu,
                        dispositivo=nombre, epocas=args.epocas, val_accuracy=round(acc_gpu, 4))
    print(f"[resultado] speedup = {np.mean(t_cpu) / np.mean(t_gpu):.2f}x")


main()
