#!/usr/bin/env bash
# Evidencia de ejecucion en Linux. Correr dentro de `nix develop`.
#
# Levanta un compositor Wayland sin monitor (sway headless, 1920x1080) para no depender
# de lo que se ve en pantalla. Ahi ejecuta las fases 0-4, cada una en su terminal (foot),
# con btop (uso por nucleo de CPU) y nvidia-smi (uso/memoria de GPU) abiertos al lado,
# y toma capturas con grim en los momentos clave. Guarda en $P3_RESULTADOS:
#   logs/NN_*.log     salida completa de cada script
#   capturas/*.png    capturas de pantalla
#   *.json, figuras/  tiempos y graficas (mismo formato que resultados/)
#
# Uso:  ASL_DATA=/ruta/asl_alphabet_train scripts/capturar_evidencia.sh
#       FASES="2 3" scripts/capturar_evidencia.sh    # repetir solo algunas fases
set -euo pipefail
cd "$(dirname "$0")/.."

export P3_RESULTADOS=${P3_RESULTADOS:-$PWD/resultados/rtx5070}
export PYTHONUNBUFFERED=1
export FASES=${FASES:-0 1 2 3}
CAP=$P3_RESULTADOS/capturas
LOG=$P3_RESULTADOS/logs

# --- fuera del compositor: lanzar sway headless que vuelve a ejecutar este script ---
if [ -z "${P3_DENTRO_SWAY:-}" ]; then
  mkdir -p "$CAP" "$LOG"
  for f in $FASES; do rm -f "$LOG"/0${f}_*.log "$CAP"/0${f}_*.png; done
  rm -f "$LOG"/.cerrar_* "$LOG"/sesion.log
  conf=$(mktemp)
  cat > "$conf" <<EOF
output HEADLESS-1 resolution 1920x1080 bg #1e1e1e solid_color
default_border pixel 2
font monospace 10
exec sh -c 'P3_DENTRO_SWAY=1 $PWD/scripts/capturar_evidencia.sh > $LOG/sesion.log 2>&1; swaymsg exit'
EOF
  env -u DISPLAY -u WAYLAND_DISPLAY -u HYPRLAND_INSTANCE_SIGNATURE \
    WLR_BACKENDS=headless WLR_RENDERER=pixman WLR_LIBINPUT_NO_DEVICES=1 \
    sway -c "$conf" >/dev/null 2>&1
  rm -f "$conf"
  cat "$LOG/sesion.log"
  exit 0
fi

# --- dentro de sway headless ---
ventana() {   # comando
  foot --font=monospace:size=10 bash -c "$1" >/dev/null 2>&1 &
  sleep 2
}
cerrar() { touch "$LOG/.cerrar_$1"; sleep 1.5; }   # la terminal sale al ver la marca
esperar() {   # log, patron
  until grep -q -- "$2" "$1" 2>/dev/null; do sleep 0.5; done
}
captura() {   # nombre, [espera en s]
  sleep "${2:-0}"
  grim "$CAP/$1.png"
  echo "[captura] $CAP/$1.png"
}
fase() {      # id, script, log
  ventana "python -u src/$2 2>&1 | tee $LOG/$3; echo __FIN__ >> $LOG/$3; until [ -f $LOG/.cerrar_$1 ]; do sleep 0.5; done"
}

contar() { grep -c -- "$2" "$1" 2>/dev/null || true; }
esperar_n() { # log, patron, n: espera a que el patron aparezca n veces
  until [ "$(contar "$1" "$2")" -ge "$3" ]; do sleep 0.3; done
}
corre() { [[ " $FASES " == *" $1 "* ]]; }

# 0. Hardware reconocido (nvidia-smi + TensorFlow + Numba)
if corre 0; then
  fase hw 00_verificar_hardware.py 00_hardware.log
  esperar "$LOG/00_hardware.log" __FIN__
  captura 00_hardware_nvidia_smi 1
  cerrar hw
fi

# 1. Preparacion: secuencial (1 nucleo) vs Pool (todos los nucleos), con btop al lado
if corre 1; then
  ventana "btop -u 500"
  fase f1 01_preparacion.py 01_preparacion.log
  esperar "$LOG/01_preparacion.log" "secuencial corrida 1/"
  captura 01_cpu_secuencial 1
  esperar "$LOG/01_preparacion.log" "paralelo   corrida 1/"
  captura 01_cpu_paralelo_pool 0.7
  esperar "$LOG/01_preparacion.log" __FIN__
  captura 01_salida_preparacion 1
  cerrar f1
  pkill -f "btop -u 500" || true; sleep 1
fi

# 2. Entrenamiento: CPU (con btop) y luego GPU (con nvidia-smi en vivo)
if corre 2; then
  L=$LOG/02_entrenamiento.log
  ventana "btop -u 500"
  fase f2 02_entrenamiento.py 02_entrenamiento.log
  esperar_n "$L" "Epoch 3/5" 1          # a mitad de la 1a repeticion en CPU
  captura 02_entrenamiento_cpu 2
  esperar "$L" "variante paralela"
  pkill -f "btop -u 500" || true; sleep 1
  ventana "watch -n 0.5 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv; nvidia-smi'"
  esperar_n "$L" "Epoch 3/5" 5          # GPU, repeticion 2, en plena epoca
  captura 02_entrenamiento_gpu_nvidia_smi 0.8
  esperar "$L" __FIN__
  captura 02_salida_entrenamiento 1
  cerrar f2
  pkill -f "watch -n 0.5 nvidia-smi" || true; sleep 1
fi

# 3. Kernel de convolucion Numba CUDA (la parte GPU dura milisegundos: se captura la salida)
if corre 3; then
  fase f3 03_kernel_cuda.py 03_kernel_cuda.log
  esperar "$LOG/03_kernel_cuda.log" __FIN__
  captura 03_salida_kernel_cuda 1
  cerrar f3
fi
rm -f "$LOG"/.cerrar_*

# 4. Tabla y graficas
python src/04_comparacion.py --dir "$P3_RESULTADOS" | tee "$LOG/04_comparacion.log"
echo "[listo] evidencia en $P3_RESULTADOS"
