{
  description = "Synthia GUI — Tauri 2 (React + Vite + Rust)";

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
            nodejs_22
            rustc cargo rust-analyzer rustfmt clippy

            # Libs nativas que Tauri 2 enlaza en build
            pkg-config
            webkitgtk_4_1
            libsoup_3
            gtk3
            librsvg
            openssl
            glib
          ];

          shellHook = ''
            echo "▸ synthia/gui devShell — node $(node --version), rust $(rustc --version | cut -d' ' -f2)"
          '';
        };
      });
}
