#!/bin/sh
set -eu

# PyJulia cannot embed Julia in uv's statically linked CPython. Start Julia
# first via python-jl, then run Python with the repository's Julia environment.
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export JULIA_PROJECT="$root/.julia-env"
export JULIA_LOAD_PATH="@:@stdlib"
export JULIA_DEPOT_PATH="$root/.julia-depot:"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"

# python-jl starts from Julia, bypassing uv Python's library RPATH. Restore it
# so extension modules such as _tkinter load their matching bundled libraries.
python_lib=$(uv run --project "$root" python -c \
    'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export LD_LIBRARY_PATH="$python_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec uv run --project "$root" python-jl "$@"
