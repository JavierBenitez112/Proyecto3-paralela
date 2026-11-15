"""
Fase "convolucion": kernel CUDA escrito a mano (numba.cuda) para la operacion dominante
de la CNN, una convolucion 2D 3x3 con padding "same", un canal, sobre un lote de imagenes.

Distribucion del trabajo: UN HILO POR PIXEL DE SALIDA. Los hilos se agrupan en bloques
de (4, 8, 8) = 256 hilos y los bloques forman un grid 3D que cubre (N, 64, 64).

Lee    : x_train de cfg.RUTA_TENSORES (o --tensores PATH)
Hace   : convolucion secuencial (bucles Python) vs kernel CUDA; mide por separado
         transferencia CPU->GPU, kernel y transferencia GPU->CPU; verifica con allclose
Escribe: resultados/convolucion_secuencial.json y resultados/convolucion_paralelo.json
Ejecutar: python src/03_kernel_cuda.py [--n-imagenes 2000] [--tensores PATH]
"""

import argparse
import math
import time

import numpy as np

import config as cfg

FILTRO = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 4  # tipo Sobel
HILOS_POR_BLOQUE = (4, 8, 8)


def conv2d_secuencial(imgs, filtro):
    """Convolucion 3x3 same con bucles sobre n, y, x. Lenta a proposito (linea base)."""
    n_img, alto, ancho = imgs.shape
    salida = np.zeros_like(imgs)
    for n in range(n_img):
        for y in range(alto):
            for x in range(ancho):
                acc = np.float32(0)
                for dy in range(3):
                    for dx in range(3):
                        yy, xx = y + dy - 1, x + dx - 1
                        if 0 <= yy < alto and 0 <= xx < ancho:
                            acc += imgs[n, yy, xx] * filtro[dy, dx]
                salida[n, y, x] = acc
    return salida


try:
    from numba import cuda
except ImportError:
    cuda = None


def conv2d_kernel(imgs, filtro, salida):
    # cada hilo calcula UN pixel de salida: (imagen n, fila y, columna x)
    n, y, x = cuda.grid(3)
    if n >= imgs.shape[0] or y >= imgs.shape[1] or x >= imgs.shape[2]:
        return
    acc = 0.0
    for dy in range(3):
        for dx in range(3):
            yy = y + dy - 1
            xx = x + dx - 1
            if 0 <= yy < imgs.shape[1] and 0 <= xx < imgs.shape[2]:
                acc += imgs[n, yy, xx] * filtro[dy, dx]
    salida[n, y, x] = acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-imagenes", type=int, default=2000)
    ap.add_argument("--tensores", default=cfg.RUTA_TENSORES)
    args = ap.parse_args()

    datos = np.load(args.tensores, allow_pickle=False)
    imgs = np.ascontiguousarray(datos["x_train"][:args.n_imagenes, :, :, 0], dtype=np.float32)
    n = len(imgs)
    print(f"[cargado]  {n} imagenes de {imgs.shape[1]}x{imgs.shape[2]}")

    print("[info] variante secuencial (bucles Python)")
    tiempos_seq = []
    for i in range(cfg.REPETICIONES):
        inicio = time.perf_counter()
        ref = conv2d_secuencial(imgs, FILTRO)
        tiempos_seq.append(time.perf_counter() - inicio)
        print(f"[info] repeticion {i + 1}/{cfg.REPETICIONES}: {tiempos_seq[-1]:.2f} s")
    cfg.guardar_tiempos("convolucion", "secuencial", tiempos_seq, n_imagenes=n)

    if cuda is None or not cuda.is_available():
        print("[aviso] no hay GPU CUDA disponible, se omite la variante paralela")
        return

    # el simulador (NUMBA_ENABLE_CUDASIM=1) no tiene get_current_device
    nombre = cuda.get_current_device().name if hasattr(cuda, "get_current_device") else "simulador CUDA"
    print(f"[hardware] GPU: {nombre}")
    kernel = cuda.jit(conv2d_kernel)

    bloques = tuple(math.ceil(d / t) for d, t in zip(imgs.shape, HILOS_POR_BLOQUE))
    total = math.prod(bloques) * math.prod(HILOS_POR_BLOQUE)
    print(f"[grid] hilos por bloque {HILOS_POR_BLOQUE} ({math.prod(HILOS_POR_BLOQUE)} hilos/bloque)")
    print(f"[grid] bloques por grid {bloques} ({math.prod(bloques)} bloques)")
    print(f"[grid] hilos totales lanzados {total} (pixeles de salida: {imgs.size})")

    # calentamiento sin medir: compilacion JIT del kernel y contexto CUDA
    d_imgs, d_filtro = cuda.to_device(imgs), cuda.to_device(FILTRO)
    d_sal = cuda.device_array_like(imgs)
    kernel[bloques, HILOS_POR_BLOQUE](d_imgs, d_filtro, d_sal)
    cuda.synchronize()

    ida, ker, vuelta = [], [], []
    for i in range(cfg.REPETICIONES):
        t0 = time.perf_counter()
        d_imgs, d_filtro = cuda.to_device(imgs), cuda.to_device(FILTRO)
        d_sal = cuda.device_array_like(imgs)
        cuda.synchronize()
        t1 = time.perf_counter()
        kernel[bloques, HILOS_POR_BLOQUE](d_imgs, d_filtro, d_sal)
        cuda.synchronize()
        t2 = time.perf_counter()
        res = d_sal.copy_to_host()
        t3 = time.perf_counter()
        ida.append(t1 - t0)
        ker.append(t2 - t1)
        vuelta.append(t3 - t2)
        print(f"[info] repeticion {i + 1}/{cfg.REPETICIONES}: ida {ida[-1]:.4f} s, "
              f"kernel {ker[-1]:.4f} s, vuelta {vuelta[-1]:.4f} s")

    ok = np.allclose(res, ref, atol=1e-5)
    print(f"[verificacion] GPU == secuencial (allclose atol=1e-5): {ok}")
    if not ok:
        raise SystemExit("[error] el resultado de la GPU no coincide con el secuencial")

    prom = lambda v: sum(v) / len(v)
    totales = [a + b + c for a, b, c in zip(ida, ker, vuelta)]
    cfg.guardar_tiempos(
        "convolucion", "paralelo", totales, n_imagenes=n,
        transferencia_ida_seg=round(prom(ida), 4), kernel_seg=round(prom(ker), 4),
        transferencia_vuelta_seg=round(prom(vuelta), 4),
        hilos_por_bloque=list(HILOS_POR_BLOQUE), bloques_por_grid=list(bloques),
    )
    print(f"[resultado] speedup total = {prom(tiempos_seq) / prom(totales):.1f}x, "
          f"solo kernel = {prom(tiempos_seq) / prom(ker):.1f}x")


main()
