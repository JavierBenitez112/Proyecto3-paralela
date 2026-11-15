# Proyecto 3: Mejorando algoritmos con CUDA

Aceleración del laboratorio de lenguaje de señas (ASL, 29 clases x 600 imágenes = 17,400 imágenes de 64x64 en grises)
combinando paralelismo en CPU (`multiprocessing`) y en GPU (TensorFlow/cuDNN y un kernel propio en Numba CUDA).
Cada fase mide la versión secuencial y la paralela con las mismas condiciones (5 repeticiones, semilla 123).

## Estructura

```
src/config.py            rutas y parámetros (fuente única) + guardar_tiempos
src/01_preparacion.py    FASE 1: carga/resize de imágenes, secuencial vs Pool (CPU)
src/02_entrenamiento.py  FASE 2: CNN en /CPU:0 vs /GPU:0 (CUDA/cuDNN)
src/03_kernel_cuda.py    FASE 3: convolución 3x3, bucles NumPy vs kernel Numba CUDA
src/04_comparacion.py    tabla comparativa y gráficas
notebooks/proyecto3_colab.ipynb   runner para Google Colab (T4)
docs/informe.md          borrador del informe
```

## Ejecución local (solo partes de CPU)

```
pip install -r requirements.txt
export ASL_DATA=/ruta/a/asl_alphabet_train   # carpeta que contiene A, B, ..., nothing, space
python src/01_preparacion.py
python src/04_comparacion.py
```

Las fases con GPU (2 y 3) requieren una GPU NVIDIA con CUDA.

## Ejecución en una laptop con GPU NVIDIA (ej. RTX 3050 Ti)

En Windows hay que usar **WSL2**: TensorFlow >= 2.11 no usa la GPU en Windows nativo.
En Linux nativo se salta el paso 1.

1. Driver de NVIDIA actualizado en Windows, luego en PowerShell (admin): `wsl --install` y reiniciar.
   No se instala CUDA a mano: el driver de Windows expone la GPU a WSL y las librerías CUDA llegan por pip.
2. Dentro de Ubuntu (WSL), con el repo y el dataset copiados:

```
sudo apt update && sudo apt install -y python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt "tensorflow[and-cuda]"
export ASL_DATA=/ruta/a/asl_alphabet_train
python src/00_verificar_hardware.py   # debe listar la GPU en nvidia-smi, TensorFlow y Numba
python src/01_preparacion.py
python src/02_entrenamiento.py        # escribe resultados/nvidia_smi_log.csv
python src/03_kernel_cuda.py
python src/04_comparacion.py
```

Copiar el dataset dentro de WSL (`~/...`), no leerlo desde `/mnt/c/`: el acceso a disco de Windows desde WSL
es lento y distorsiona la fase 1. Para las capturas: `watch -n 1 nvidia-smi` en otra terminal durante la fase 2
y el Administrador de tareas (pestaña Rendimiento > CPU, vista de procesadores lógicos) durante la fase 1.

## Ejecución en Colab (GPU)

Abrir `notebooks/proyecto3_colab.ipynb`, elegir Runtime > Change runtime type > T4 GPU y ejecutar las celdas en orden.
El notebook descarga el dataset de Kaggle (`grassknoted/asl-alphabet`) y define `ASL_DATA`.

## Orden de los scripts

1. `01_preparacion.py` (genera `data/processed/tensores.npz`)
2. `02_entrenamiento.py`
3. `03_kernel_cuda.py`
4. `04_comparacion.py`

## Resultados

- `resultados/<fase>_<variante>.json`: tiempos por repetición y promedio
- `resultados/tabla_comparativa.md`: tabla secuencial vs paralelo con speedup
- `resultados/figuras/`: gráficas de tiempos, speedup y desglose transferencia/kernel
- `resultados/nvidia_smi_log.csv`: utilización de GPU durante el entrenamiento (Colab)
