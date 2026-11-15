# Proyecto 3: Mejorando algoritmos con CUDA

Informe (borrador). Todo lo marcado como **[PENDIENTE: ...]** requiere una medición o captura real.

## 1. Análisis de la solución secuencial

El laboratorio original (Laboratorio 3, clasificación de lenguaje de señas ASL) es un pipeline de scripts de Python:

- `01_particiones.py`: lista los jpg del dataset crudo (29 carpetas de 3000 imágenes), toma 600 por clase al azar (semilla 123) y las reparte de forma estratificada en train/val/test (70/15/15). Solo produce un inventario CSV de rutas.
- `02_preprocesamiento.py`: recorre el inventario imagen por imagen, la abre, la pasa a grises, la reescala de 200x200 a 64x64 con LANCZOS y la normaliza a [0, 1]. Guarda los tensores en un `.npz`.
- `03_augmentation.py`: genera copias de entrenamiento con rotación, zoom, desplazamiento, brillo y contraste (sin espejos).
- `modelado.py`: entrena y evalúa modelos Keras (CNN, red densa) y sklearn con un protocolo común (semilla, épocas máximas, EarlyStopping, accuracy en validación).

### Operaciones identificadas (de recibir las imágenes al resultado)

| No. | Operación identificada | Descripción de su funcionamiento |
|---|---|---|
| 1 | Listar archivos | `glob("*.jpg")` por clase; devuelve las rutas ordenadas de las 29 carpetas. |
| 2 | Submuestreo y partición | Elige 600 imágenes por clase sin reemplazo, las baraja y las asigna a train (420), val (90) y test (90). |
| 3 | Abrir JPG | `PIL.Image.open` lee y decodifica el archivo JPEG de 200x200 RGB a memoria. |
| 4 | Conversión a grises | `convert("L")` combina los 3 canales en uno por píxel. |
| 5 | Reescalado LANCZOS | `resize((64, 64), LANCZOS)`: cada píxel de salida es un promedio ponderado de una vecindad de la imagen original. |
| 6 | Normalización | Se divide entre 255 para dejar los píxeles en [0, 1] (float32). |
| 7 | Codificación de etiquetas | La clase (texto) se convierte en un índice entero según `CLASES`. |
| 8 | Apilado de tensores | Cada imagen se escribe en un arreglo `(N, 64, 64, 1)`; se guarda por partición con sus etiquetas. |
| 9 | Aumento de datos | Transformaciones aleatorias por lotes sobre train; se concatenan con los originales. |
| 10 | Convolución (forward) | Cada filtro 3x3 se desliza sobre el mapa de entrada; en cada posición suma productos con los pesos y el sesgo, y aplica ReLU. |
| 11 | Pooling | MaxPool 2x2 reduce a la mitad alto y ancho, quedándose con el máximo de cada ventana. |
| 12 | Capas densas | Flatten y Dense(128): multiplicación matriz-vector/matriz; Dense(29) con softmax produce probabilidades por clase. |
| 13 | Pérdida | Entropía cruzada entre las probabilidades y la etiqueta real. |
| 14 | Retropropagación | Se calculan los gradientes de la pérdida respecto a cada peso, capa por capa hacia atrás. |
| 15 | Actualización de pesos | Adam usa los gradientes para ajustar los pesos en cada lote. |
| 16 | Validación | Se repite el forward sobre val al final de cada época para medir la accuracy. |
| 17 | Predicción | `argmax` sobre las 29 probabilidades da la clase predicha. |

### Operaciones repetitivas y cantidad de repeticiones

Arquitectura del proyecto: 3 x [Conv2D 3x3 (32, 64, 64 filtros) + MaxPool], Flatten, Dense 128, Dense 29. Particiones sin aumento: 29 x 420 = 12,180 train; 29 x 90 = 2,610 val; 2,610 test.

| Operación repetitiva | Datos que procesa | Cantidad de repeticiones |
|---|---|---|
| Preprocesamiento de imagen (abrir, grises, resize, normalizar) | 1 imagen JPG 200x200 | 17,400 (29 x 600) |
| Lotes por época | train | ceil(12,180 / 64) = 191 lotes/época (x 5 épocas = 955 lotes en el benchmark) |
| Convolución capa 1 (1 -> 32 filtros, entrada 64x64) | 1 imagen | 64 x 64 x 32 = 131,072 posiciones de salida; 64x64x32x9x1 = 1,179,648 multiplicaciones-suma |
| Convolución capa 2 (32 -> 64 filtros, entrada 32x32) | 1 imagen | 32 x 32 x 64 = 65,536 posiciones; 32x32x64x9x32 = 18,874,368 mult-suma |
| Convolución capa 3 (64 -> 64 filtros, entrada 16x16) | 1 imagen | 16 x 16 x 64 = 16,384 posiciones; 16x16x64x9x64 = 9,437,184 mult-suma |
| Total convolución | 1 imagen | 29,491,200 mult-suma por imagen en el forward (solo capas conv) |
| Convolución por época (forward, capas conv) | 12,180 imágenes | 12,180 x 29,491,200 = 359,202,816,000 mult-suma (≈3.6e11) |
| Kernel de convolución del proyecto (fase 3) | N imágenes de 64x64 | N x 64 x 64 hilos; para N = 2000: 8,192,000 hilos (cada uno 9 mult-suma) |

Nota: el backward cuesta aproximadamente el doble del forward, por lo que una época completa es varias veces mayor que la cifra anterior.

## 2. Medición secuencial

**Procedimiento.** Cada fase se mide con `time.perf_counter()`, 5 repeticiones con los mismos datos y la misma semilla (123), y se reporta el promedio y la desviación estándar. Los tiempos se guardan en `resultados/<fase>_secuencial.json`. En entrenamiento se fijan 5 épocas para que ambas variantes hagan el mismo trabajo.

| Fase | Tiempo por repetición (s) | Promedio (s) | Desv. est. (s) |
|---|---|---|---|
| Preparación (17,400 imágenes) | **[PENDIENTE: 5 tiempos]** | **[PENDIENTE]** | **[PENDIENTE]** |
| Entrenamiento CPU (5 épocas) | **[PENDIENTE: 5 tiempos]** | **[PENDIENTE]** | **[PENDIENTE]** |
| Convolución NumPy (N = 2000) | **[PENDIENTE: 5 tiempos]** | **[PENDIENTE]** | **[PENDIENTE]** |

Equipo: **[PENDIENTE: CPU, núcleos, RAM, sistema operativo]**.

**Factores que afectan el rendimiento medido:**
- E/S de disco: la lectura de 17,400 archivos depende del disco y de si el sistema los tiene en caché (la primera repetición suele ser más lenta).
- Decodificación JPEG: es trabajo de CPU por imagen y domina el costo de la preparación.
- GIL de Python: los hilos no aceleran código Python puro, por eso se usan procesos.
- Caché de CPU y acceso a memoria en los bucles anidados.
- Thermal throttling: la CPU reduce frecuencia al calentarse en corridas largas.
- Procesos en segundo plano que compiten por CPU y memoria.

## 3. Oportunidades de paralelización

- **Preparación de imágenes (CPU multiproceso).** Cada imagen se procesa de forma independiente (sin dependencias entre ellas), así que es un problema de tareas independientes. El costo es de E/S y decodificación JPEG, trabajo irregular y de control que no se adapta bien a una GPU; se reparte entre todos los núcleos con `multiprocessing.Pool`.
- **Convolución y multiplicación de matrices (GPU).** Son aritmética idéntica repetida sobre millones de elementos (decenas de millones de mult-suma por imagen), con cada posición de salida independiente de las demás. Es el patrón de paralelismo de datos que una GPU ejecuta con miles de hilos simultáneos.
- **Se mantiene secuencial:** el bucle de épocas y lotes (cada actualización de pesos depende de la anterior), el orden de las capas dentro del forward/backward (cada capa necesita la salida de la anterior), la partición y el listado de archivos (costo mínimo) y la escritura final de resultados. Por la ley de Amdahl, estas partes limitan la ganancia máxima.

## 4. Diseño del algoritmo paralelo

**Kernel de convolución (Numba CUDA).** Se usa un hilo por píxel de salida de un volumen `(N, 64, 64)`. Cada hilo obtiene su índice con `cuda.grid(3)` = (imagen, fila, columna), verifica límites (`if i < N and y < 64 and x < 64`) y calcula la suma de 9 productos del filtro 3x3 con relleno "same" (los vecinos fuera de la imagen valen 0).

- Hilos por bloque: (4, 8, 8) = 256.
- Bloques por grid: ceil(dim / tpb) en cada eje. Para N = 2000: (ceil(2000/4), ceil(64/8), ceil(64/8)) = (500, 8, 8) = 32,000 bloques.
- Hilos totales: 32,000 x 256 = 8,192,000 = 2000 x 64 x 64 (uno por píxel, sin hilos sobrantes en este caso). La verificación de límites protege cuando N no es múltiplo de 4.

```
grid = (500, 8, 8) bloques            bloque = (4, 8, 8) hilos
 +----+----+--...--+                  eje0: 4 imagenes
 | b0 | b1 |       |  eje y: 8 bloques    eje1: 8 filas
 +----+----+--...--+  eje x: 8 bloques    eje2: 8 columnas
 ... x 500 en el eje de imagenes          = 256 hilos
hilo (i, y, x) -> salida[i, y, x] = suma de 9 vecinos * filtro
```

**Flujo de datos:** host (NumPy) -> `cuda.to_device` (transferencia CPU->GPU) -> kernel en la GPU -> `copy_to_host` (GPU->CPU). Se miden por separado la ida, el kernel y la vuelta, con `cuda.synchronize()` antes de leer el reloj porque el lanzamiento del kernel es asíncrono.

**Equivalente en TensorFlow/cuDNN.** Al entrenar en `/GPU:0`, TensorFlow envía los tensores del lote a la memoria de la GPU y cada capa se ejecuta con kernels de cuDNN (convolución, pooling) y cuBLAS (capas densas), que usan el mismo principio de hilos/bloques, con algoritmos más optimizados (memoria compartida, Winograd/FFT, Tensor Cores según la GPU). El forward y el backward corren en la GPU y solo regresan métricas al host.

## 5. Implementación y medición paralela

**Cómo ejecutar:** abrir `notebooks/proyecto3_colab.ipynb` en Google Colab con GPU T4 y correr las celdas en orden (ver README).

**Librerías y su rol:**
- `multiprocessing` (Pool): reparte las imágenes entre los núcleos de la CPU (fase 1).
- TensorFlow + CUDA + cuDNN: construye la CNN y la ejecuta en `/CPU:0` o `/GPU:0` (fase 2).
- Numba CUDA: kernel de convolución escrito a mano con control explícito de hilos, bloques y transferencias (fase 3).
- NumPy, Pillow, matplotlib: datos, lectura de imágenes y gráficas.

| Fase | Tiempo por repetición (s) | Promedio (s) | Desv. est. (s) |
|---|---|---|---|
| Preparación, Pool de **[PENDIENTE: n núcleos]** procesos | **[PENDIENTE]** | **[PENDIENTE]** | **[PENDIENTE]** |
| Entrenamiento GPU (5 épocas) | **[PENDIENTE]** | **[PENDIENTE]** | **[PENDIENTE]** |
| Convolución Numba CUDA (N = 2000) | **[PENDIENTE]** | **[PENDIENTE]** | **[PENDIENTE]** |

| Convolución CUDA | Tiempo (s) |
|---|---|
| Transferencia CPU -> GPU | **[PENDIENTE]** |
| Kernel | **[PENDIENTE]** |
| Transferencia GPU -> CPU | **[PENDIENTE]** |

GPU usada: **[PENDIENTE: modelo, VRAM, versión de CUDA según `nvidia-smi`]**. Accuracy de validación CPU vs GPU: **[PENDIENTE]**.

## 6. Análisis y conclusiones

**Tabla comparativa:** **[PENDIENTE: pegar `resultados/tabla_comparativa.md`]**. Figuras en `resultados/figuras/` (`tiempos_por_fase.png`, `speedup_por_fase.png`, `desglose_gpu.png`): **[PENDIENTE: insertar]**.

Speedup: **S = T_secuencial / T_paralelo**. Un valor mayor a 1 indica que la versión paralela es más rápida.

**(a) ¿Por qué procesar imágenes/archivos en varios núcleos de CPU y delegar solo la matemática de la red a la GPU?** Preparar una imagen es leer un archivo, decodificar JPEG y reescalar: mucha E/S y lógica de control, con tareas independientes y de duración irregular. Eso se reparte bien entre unos pocos núcleos potentes de CPU, y una GPU no tiene acceso directo al sistema de archivos ni es eficiente decodificando JPEG. En cambio, la convolución y las multiplicaciones de matrices son la misma operación aritmética sobre millones de datos, que es lo que una GPU hace mejor. Cada procesador se usa para lo que mejor hace, y los tensores ya preparados se envían a la GPU una sola vez.

**(b) ¿Cómo redujo o cambió CUDA el tiempo de procesamiento?** Reducción medida: **[PENDIENTE: speedup de entrenamiento y de convolución]**. Mecanismo: en lugar de calcular una posición de salida tras otra, la GPU lanza miles de hilos que calculan posiciones distintas a la vez, y su ancho de banda de memoria y cantidad de unidades aritméticas son mucho mayores que los de la CPU. En el entrenamiento, cuDNN además usa kernels optimizados. El costo es que aparecen tiempos nuevos (transferencias host-device, lanzamiento) que la versión secuencial no tiene.

**(c) Factores que justifican o limitan el uso de CUDA:**
- Sobrecosto de transferencia: copiar datos por PCIe puede costar más que el cálculo si el trabajo por dato es poco.
- Lotes o problemas pequeños: no ocupan la GPU, y el sobrecosto fijo domina.
- Ley de Amdahl: la parte secuencial (carga de datos, bucle de épocas) acota el speedup total: S <= 1 / ((1 - p) + p/s), con p la fracción paralelizable y s la aceleración de esa fracción.
- VRAM limitada: el conjunto de datos y el modelo deben caber en memoria de la GPU (o procesarse por lotes).
- Disponibilidad de hardware: requiere una GPU NVIDIA y drivers/toolkit compatibles; en Colab el acceso es limitado.
- Calentamiento/JIT: la primera llamada incluye inicialización de CUDA y compilación de Numba; conviene excluirla o reportarla aparte. **[PENDIENTE: valor medido de la primera ejecución si se mide]**.

### Capturas requeridas por la rúbrica

- [ ] Uso de los núcleos de CPU durante la fase 1 (panel de recursos de Colab o `top`). **[PENDIENTE: captura]**
- [ ] `nvidia-smi` durante la fase 2 mostrando la GPU en uso (y/o `resultados/nvidia_smi_log.csv`). **[PENDIENTE: captura]**
- [ ] `nvidia-smi` inicial con el modelo de GPU. **[PENDIENTE: captura]**
- [ ] Salida de cada script con los tiempos. **[PENDIENTE: captura]**
