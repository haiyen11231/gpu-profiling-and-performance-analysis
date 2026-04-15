{
  description = "SC4064 Course Project";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          gnumake
          cudatoolkit
          stdenv.cc
          binutils
          kubectl
          k9s

          # ROCm development
          rocmPackages.clr # HIP runtime + headers
          rocmPackages.hipcc # HIP compiler wrapper
          rocmPackages.hip-common
          rocmPackages.rocm-device-libs
          rocmPackages.rocm-cmake
        ];

        # This makes the language server aware of HIP headers
        shellHook = ''
              # CUDA headers
          export C_INCLUDE_PATH="${pkgs.cudatoolkit}/include:$C_INCLUDE_PATH"
          export CPLUS_INCLUDE_PATH="${pkgs.cudatoolkit}/include:$CPLUS_INCLUDE_PATH"
          export LIBRARY_PATH="${pkgs.cudatoolkit}/lib:$LIBRARY_PATH"

          # HIP/ROCm headers
          export C_INCLUDE_PATH="${pkgs.rocmPackages.clr}/include:$C_INCLUDE_PATH"
          export CPLUS_INCLUDE_PATH="${pkgs.rocmPackages.clr}/include:$CPLUS_INCLUDE_PATH"

        '';

      };
    };
}
