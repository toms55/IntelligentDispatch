{
  description = "Dev environment for the battery cycle optimisation DRL project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  };

  outputs = { self, nixpkgs, ... }@inputs:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

      # PyPI wheels expect an FHS system; NixOS has no /usr/lib.
      wheelLibs = with pkgs; [ stdenv.cc.cc.lib zlib zstd openssl ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        # git-lfs is for `scripts/publishSpace.sh`. A Hugging Face Space refuses a
        # binary file that git tracks directly, and the upload carries the agent
        # weights and the per-fold parquet.
        packages = [ pkgs.python311 pkgs.uv pkgs.git-lfs ];

        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath wheelLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          export UV_PYTHON="${pkgs.python311}/bin/python3.11"
          export UV_PYTHON_DOWNLOADS=never

          echo "Currently using:"
          python --version
          echo "Run 'uv sync' to install dependencies into .venv"
        '';
      };
    };
}
