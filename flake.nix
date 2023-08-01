{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonpkgs = pkgs.python310Packages;

        python_with_pkgs = (pythonpkgs.python.withPackages (_: with pythonpkgs; [
          python
          ipython
          lxml
          rasterio
          gdal
          pytest
          flake8
          matplotlib
          sphinx-book-theme
          sphinx
          numpydoc
          myst-parser
          #myst-nb
          sphinxcontrib-programoutput
          sphinx-autodoc-typehints
          sphinx-design

        ])).overrideAttrs (prev: {
          pname = "python";
        });

        packages = builtins.listToAttrs (map (pkg: { name = pkg.pname; value = pkg; }) ( with pkgs; [
          python_with_pkgs
          pre-commit
        ]));

      in
      {
        inherit packages;
        defaultPackage = packages.python;

        devShell = pkgs.mkShell {
          name = "variete";
          buildInputs = pkgs.lib.attrValues packages;
        };
      }

    );
}
