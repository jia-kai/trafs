#!/bin/bash -e

# setup the extra dependencies (which require custom compilation)

if ! julia -e 'using MPBNGCInterface' > /dev/null 2>&1; then
    echo "MPBNGCInterface.jl not present, installing..."
    path=$(readlink -f $(dirname $0)/third-party/MPBNGCInterface.jl)
    [ -f $path/Project.toml ] || git submodule update --init --recursive
    julia -e "using Pkg; Pkg.develop(path=\"$path\"); \
        Pkg.build(); Pkg.precompile(); Pkg.instantiate()"
fi

if ! python -c 'import julia; julia.install()'; then
    echo "Failed to install pyjulia"
    echo "Please install the julia package(with pip) or the pyjulia package(with conda)"
    exit 1
fi

if ! python -c 'import piqptr'; then
    echo "Building the customized PIQP solver"
    cd third-party/piqp
    python setup.py build
    python setup.py install
fi
