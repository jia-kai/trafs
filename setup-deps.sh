#!/bin/bash -e

# setup the extra dependencies (which require custom compilation)

root=$(readlink -f "$(dirname "$0")")
export JULIA_PROJECT="$root/.julia-env"
export JULIA_LOAD_PATH="@:@stdlib"
export JULIA_DEPOT_PATH="$root/.julia-depot:"
mkdir -p "$JULIA_PROJECT" "$root/.julia-depot"

if ! julia -e 'using MPBNGCInterface' > /dev/null 2>&1; then
    echo "MPBNGCInterface.jl not present, installing..."
    path=$(readlink -f $(dirname $0)/third-party/MPBNGCInterface.jl)
    [ -f $path/Project.toml ] || git submodule update --init --recursive
    julia -e "using Pkg; Pkg.develop(path=\"$path\"); \
        Pkg.build(); Pkg.precompile(); Pkg.instantiate()"
fi

if ! uv run --project "$root" python -c 'import julia; julia.install()'; then
    echo "Failed to install pyjulia"
    exit 1
fi

if ( ! uv run --project "$root" python -c 'import piqptr' ) || [ "$1" = "--piqp" ] ; then
    echo "Building the customized PIQP solver"
    uv pip install --project "$root" -e "$root/third-party/piqp"
fi
