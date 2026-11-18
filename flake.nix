{
  description = "Proyecto 3 - ASL con multiprocessing (CPU) y CUDA (TensorFlow + Numba)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python312;

      # Las ruedas de pip (tensorflow[and-cuda], numba-cuda) traen sus propias
      # librerias CUDA; solo necesitan libstdc++/zlib y el libcuda.so del driver.
      libs = with pkgs; [ stdenv.cc.cc.lib zlib libGL glib ];
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          python
          pkgs.pandoc
          pkgs.typst
          pkgs.grim   # capturas (scripts/capturar_evidencia.sh)
          pkgs.sway   # compositor headless para las capturas
          pkgs.foot
        ];

        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath libs}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          export PATH=/run/current-system/sw/bin:$PATH   # nvidia-smi del sistema
          export TF_CPP_MIN_LOG_LEVEL=2
          export ASL_DATA=''${ASL_DATA:-$PWD/data/raw/asl_alphabet_train}

          if [ ! -d .venv ]; then
            echo "[nix] creando .venv e instalando dependencias (una sola vez)..."
            ${python}/bin/python -m venv .venv
            .venv/bin/pip install -q --upgrade pip
            .venv/bin/pip install -q -r requirements.txt
          fi
          source .venv/bin/activate
          # dependencias cruzadas entre ruedas nvidia-* (p. ej. cusolver -> nvJitLink)
          for d in .venv/lib/python3.12/site-packages/nvidia/*/lib; do
            LD_LIBRARY_PATH=$PWD/$d:$LD_LIBRARY_PATH
          done
          export LD_LIBRARY_PATH
          echo "[nix] entorno listo. ASL_DATA=$ASL_DATA"
          echo "      python src/00_verificar_hardware.py   |   nix run .#informe  (PDF)"
        '';
      };

      # nix run .#informe -> docs/informe.pdf
      apps.${system}.informe = {
        type = "app";
        program = toString (pkgs.writeShellScript "make-informe" ''
          set -e
          cd docs
          ${pkgs.pandoc}/bin/pandoc informe.md -o informe.pdf \
            --pdf-engine=${pkgs.typst}/bin/typst \
            --resource-path=.:.. --pdf-engine-opt=--root=.. \
            -V mainfont="Libertinus Serif" -V papersize=us-letter -V margin.x=2cm -V margin.y=2cm
          echo "docs/informe.pdf generado"
        '');
      };
    };
}
