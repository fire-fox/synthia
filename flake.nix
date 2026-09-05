{
  description = "Synthia — AI voice assistant (Python ≥3.10)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python313
            python313Packages.pip
            python313Packages.virtualenv
            ffmpeg                # synthia procesa audio
            portaudio             # PortAudio para mic/speaker
            pkg-config
          ];

          shellHook = ''
            echo "▸ synthia devShell — python $(python --version)"
            if [ ! -d venv ]; then
              echo "  primer setup: python -m venv venv && source venv/bin/activate && pip install -e ."
            elif [ -z "$VIRTUAL_ENV" ]; then
              echo "  para activar venv:  source venv/bin/activate"
            fi
          '';
        };
      });
}
