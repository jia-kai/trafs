import numpy as np

from distutils.extension import Extension
from pathlib import Path
import contextlib
import time

@contextlib.contextmanager
def setup_pyx_import():
    """a context to import .pyx cython files"""
    # use a local cache in the directory of this repo to avoid conflicts with
    # multiple source versions on the system
    build_dir = Path(__file__).resolve().parent.parent / 'build'
    import pyximport
    px = pyximport.install(
        build_dir=build_dir,
        setup_args={
            'include_dirs': [np.get_include()],
        },
        language_level=3,
    )
    yield
    pyximport.uninstall(*px)

def default_make_ext_for_pyx(modname, pyxfilename):
    return Extension(
        name=modname,
        sources=[pyxfilename],
        extra_compile_args=['-O3', '-march=native'],
        language='c++',
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )

class CPUTimer:
    """get the CPU time used by the current process"""
    _t0: int

    def __init__(self):
        self.reset()

    def reset(self):
        """reset the timer"""
        self._t0 = time.process_time_ns()

    def elapsed(self) -> float:
        """return the elapsed time since previous reset in seconds"""
        return (time.process_time_ns() - self._t0) / 1e9
