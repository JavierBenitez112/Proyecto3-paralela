"""
Paso 0: VERIFICACION DEL HARDWARE
------------------------------------------------------------
Imprime lo que el entorno reconoce: nucleos de CPU, GPU segun nvidia-smi, GPU segun
TensorFlow y GPU segun Numba. Es la evidencia de que cada fase vera su hardware, y el
primer lugar donde mirar si una fase GPU se omite.

Ejecutar:  python src/00_verificar_hardware.py
"""

import os
import platform
import shutil
import subprocess

print(f"[cpu] {platform.processor() or platform.machine()} -> {os.cpu_count()} nucleos logicos")
print(f"[so ] {platform.platform()}")

if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"])
else:
    print("[FALLO] nvidia-smi no esta en el PATH: falta el driver de NVIDIA")

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(f"[tensorflow {tf.__version__}] GPUs: {gpus}")
if not gpus:
    print("[FALLO] TensorFlow no ve la GPU. En Windows nativo TensorFlow >= 2.11 no usa GPU:")
    print("        hay que correr dentro de WSL2 con  pip install 'tensorflow[and-cuda]'")

try:
    from numba import cuda
    if cuda.is_available():
        print(f"[numba] GPU: {cuda.get_current_device().name}")
    else:
        print("[FALLO] Numba no encuentra CUDA. Diagnostico:")
        cuda.detect()
except Exception as e:  # numba ausente o librerias CUDA sin encontrar
    print(f"[FALLO] Numba CUDA: {e}")
