"""Pruebas que no necesitan GPU ni dataset: correctitud de la convolucion de
referencia, contrato de guardar_tiempos y tabla de 04_comparacion.py.

Ejecutar:  python -m pytest tests
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import config as cfg  # noqa: E402


def cargar(nombre):
    """Importa src/<nombre>.py (los nombres empiezan con digito, no sirve import)."""
    spec = importlib.util.spec_from_file_location(nombre, SRC / f"{nombre}.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def conv_vectorizada(imgs, filtro):
    """Convolucion 3x3 same con padding de ceros, sin bucles por pixel."""
    pad = np.pad(imgs, ((0, 0), (1, 1), (1, 1)))
    alto, ancho = imgs.shape[1:]
    salida = np.zeros_like(imgs)
    for dy in range(3):
        for dx in range(3):
            salida += pad[:, dy:dy + alto, dx:dx + ancho] * filtro[dy, dx]
    return salida


def test_conv2d_secuencial_coincide_con_numpy():
    kc = cargar("03_kernel_cuda")
    imgs = np.random.default_rng(0).random((3, 9, 7), dtype=np.float32)
    np.testing.assert_allclose(kc.conv2d_secuencial(imgs, kc.FILTRO),
                               conv_vectorizada(imgs, kc.FILTRO), atol=1e-6)


def test_conv2d_bordes_con_padding_cero():
    kc = cargar("03_kernel_cuda")
    imgs = np.ones((1, 4, 4), dtype=np.float32)
    salida = kc.conv2d_secuencial(imgs, kc.FILTRO)
    # el filtro tipo Sobel suma 0: en el interior de una imagen constante da 0
    np.testing.assert_allclose(salida[0, 1:-1, 1:-1], 0, atol=1e-7)
    # en los bordes izquierdo/derecho falta una columna y la respuesta es != 0
    assert salida[0, 1, 0] != 0 and salida[0, 1, -1] != 0


@pytest.fixture
def resultados_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DIR_RESULTADOS", tmp_path)
    return tmp_path


def test_guardar_tiempos_escribe_el_contrato(resultados_tmp):
    registro = cfg.guardar_tiempos("preparacion", "secuencial", [1.0, 2.0, 3.0], nucleos=1)
    en_disco = json.loads((resultados_tmp / "preparacion_secuencial.json").read_text())
    assert en_disco == registro
    assert en_disco["promedio_seg"] == 2.0
    assert en_disco["tiempos_seg"] == [1.0, 2.0, 3.0]
    assert en_disco["nucleos"] == 1


def test_guardar_tiempos_rechaza_lista_vacia(resultados_tmp):
    with pytest.raises(ValueError):
        cfg.guardar_tiempos("preparacion", "secuencial", [])


def test_comparacion_calcula_speedup(resultados_tmp, monkeypatch):
    cfg.guardar_tiempos("preparacion", "secuencial", [10.0, 10.0])
    cfg.guardar_tiempos("preparacion", "paralelo", [2.0, 2.0])
    comp = cargar("04_comparacion")
    monkeypatch.setattr(sys, "argv", ["04_comparacion.py", "--dir", str(resultados_tmp)])
    comp.main()
    tabla = (resultados_tmp / "tabla_comparativa.md").read_text(encoding="utf-8")
    assert "| preparacion | 10.000 |" in tabla and "5.00x" in tabla
    assert "| entrenamiento | pendiente |" in tabla
    assert (resultados_tmp / "figuras" / "speedup_por_fase.png").exists()
