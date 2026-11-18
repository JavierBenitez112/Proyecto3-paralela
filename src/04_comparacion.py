"""Fase 4: tabla comparativa y graficas secuencial vs paralelo.

Lee resultados/<fase>_secuencial.json y <fase>_paralelo.json. Si una fase solo
tiene una variante se marca "pendiente" y se omite de las graficas.

Ejecutar:  python src/04_comparacion.py   (opcional: --dir <carpeta con los json>)
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg

FASES = ["preparacion", "entrenamiento", "convolucion"]


def leer(carpeta, fase, variante):
    ruta = carpeta / f"{fase}_{variante}.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def desv(r):
    t = r["tiempos_seg"]
    return statistics.stdev(t) if len(t) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=cfg.DIR_RESULTADOS)
    args = ap.parse_args()
    carpeta = args.dir
    figuras = carpeta / "figuras"  # con el --dir por defecto es cfg.DIR_FIGURAS
    figuras.mkdir(parents=True, exist_ok=True)

    filas = ["| Fase | T secuencial (s) | desv | T paralelo (s) | desv | Speedup |",
             "|---|---|---|---|---|---|"]
    completas = []
    for fase in FASES:
        s, p = leer(carpeta, fase, "secuencial"), leer(carpeta, fase, "paralelo")
        if s and p:
            sp = s["promedio_seg"] / p["promedio_seg"]
            filas.append(f"| {fase} | {s['promedio_seg']:.3f} | {desv(s):.3f} | "
                         f"{p['promedio_seg']:.3f} | {desv(p):.3f} | {sp:.2f}x |")
            completas.append((fase, s, p, sp))
        else:
            filas.append(f"| {fase} | pendiente | - | pendiente | - | - |")
    tabla = "\n".join(filas)
    print(tabla)
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "tabla_comparativa.md").write_text(tabla + "\n", encoding="utf-8")

    if completas:
        nombres = [c[0] for c in completas]
        x = range(len(nombres))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([i - 0.2 for i in x], [c[1]["promedio_seg"] for c in completas], 0.4,
               yerr=[desv(c[1]) for c in completas], label="secuencial")
        ax.bar([i + 0.2 for i in x], [c[2]["promedio_seg"] for c in completas], 0.4,
               yerr=[desv(c[2]) for c in completas], label="paralelo")
        ax.set_xticks(list(x), nombres)
        ax.set_yscale("log")
        ax.set_ylabel("tiempo promedio (s, escala log)")
        ax.set_title("Tiempo por fase")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figuras / "tiempos_por_fase.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(nombres, [c[3] for c in completas], color="tab:green")
        ax.axhline(1, color="gray", ls="--")
        ax.set_ylabel("speedup = T_sec / T_par")
        ax.set_title("Speedup por fase")
        fig.tight_layout()
        fig.savefig(figuras / "speedup_por_fase.png", dpi=150)
        plt.close(fig)

    cp = leer(carpeta, "convolucion", "paralelo")
    claves = ["transferencia_ida_seg", "kernel_seg", "transferencia_vuelta_seg"]
    if cp and all(k in cp for k in claves):
        fig, ax = plt.subplots(figsize=(4, 4))
        fondo = 0
        for k, etiqueta in zip(claves, ["CPU->GPU", "kernel", "GPU->CPU"]):
            ax.bar("convolucion", cp[k], bottom=fondo, label=etiqueta)
            fondo += cp[k]
        ax.set_ylabel("tiempo (s)")
        ax.set_title("Desglose GPU")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figuras / "desglose_gpu.png", dpi=150)
        plt.close(fig)

    print(f"[guardado] tabla_comparativa.md y figuras en {figuras}")


if __name__ == "__main__":
    main()
