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
tests/                   pruebas sin GPU (python -m pytest tests)
```

## Ejecución local (solo partes de CPU)

```
pip install -r requirements.txt
export ASL_DATA=/ruta/a/asl_alphabet_train   # carpeta que contiene A, B, ..., nothing, space
python src/01_preparacion.py
python src/04_comparacion.py
```

Las fases con GPU (2 y 3) requieren una GPU NVIDIA con CUDA.

Pruebas (no necesitan GPU ni dataset): `python -m pytest tests`

## Ejecución en Windows nativo con GPU NVIDIA (configuración usada para los resultados)

Los resultados de `resultados/` se midieron así en una laptop con Ryzen 5 5600H (12 núcleos lógicos) y
RTX 3050 Ti (4 GB), sin WSL. Solo hace falta el driver de NVIDIA y Miniconda; las librerías CUDA llegan por conda/pip.

Se usan **dos entornos** porque TensorFlow y Numba no pueden compartir uno en Windows:

- TensorFlow 2.10 es la última versión con GPU en Windows nativo y exige CUDA 11.2 / cuDNN 8.1 y NumPy < 2.
- El Numba compatible con ese entorno (0.57) falla con drivers NVIDIA recientes (`access violation` al crear
  el contexto CUDA), así que la fase 3 usa el paquete `numba-cuda` con las librerías de CUDA 12.

```
conda create -y -n p3cuda -c conda-forge python=3.10 cudatoolkit=11.2 cudnn=8.1 pip
conda run -n p3cuda pip install tensorflow==2.10.1 numpy==1.23.5 pillow matplotlib pandas kagglehub

conda create -y -n p3numba -c conda-forge python=3.11 pip
conda run -n p3numba pip install "numba-cuda[cu12]" numpy cuda-core==1.2.2
```

`cuda-core` se fija en 1.2.2 porque el Control de aplicaciones de Windows bloqueó una DLL de la 1.3.0; si no
hay ese bloqueo, se puede omitir el pin.

Dataset (≈1 GB, queda en `%USERPROFILE%\.cache\kagglehub`) y ejecución, en PowerShell:

```
conda activate p3cuda
python -c "import kagglehub; print(kagglehub.dataset_download('grassknoted/asl-alphabet'))"
$env:ASL_DATA = "<ruta impresa>\asl_alphabet_train\asl_alphabet_train"   # carpeta con A, B, ..., space
python src/00_verificar_hardware.py   # debe listar la GPU en nvidia-smi y TensorFlow
python src/01_preparacion.py
python src/02_entrenamiento.py        # escribe resultados/nvidia_smi_log.csv

conda activate p3numba
python src/03_kernel_cuda.py

conda activate p3cuda
python src/04_comparacion.py
```

En `p3cuda` no se instala Numba, así que `00_verificar_hardware.py` reportará `[FALLO] Numba CUDA` ahí; es
esperado, Numba se verifica al correr la fase 3 en `p3numba`.

## Alternativa: WSL2 o Linux nativo (un solo entorno, TensorFlow actual)

En Windows con WSL2 se puede usar la última versión de TensorFlow en un único entorno.
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
- `resultados/nvidia_smi_log.csv`: utilización de GPU durante el entrenamiento

Medidos en Windows nativo (Ryzen 5 5600H, RTX 3050 Ti Laptop 4 GB):

| Fase | T secuencial (s) | T paralelo (s) | Speedup |
|---|---|---|---|
| preparacion (1 vs 12 procesos) | 17.031 | 4.466 | 3.81x |
| entrenamiento (CPU vs GPU, 5 épocas) | 170.324 | 16.162 | 10.54x |
| convolucion (bucles Python vs kernel CUDA, N = 2000) | 46.217 | 0.024 | 1950x |
