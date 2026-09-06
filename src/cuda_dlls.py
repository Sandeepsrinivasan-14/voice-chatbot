"""
Windows-only shim: gets faster-whisper's CTranslate2 backend to actually
find the CUDA cuBLAS/cuDNN DLLs.

On Linux, the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` pip wheels work
out of the box because CTranslate2's shared library carries an RPATH
pointing at them. Windows has no RPATH equivalent, and CTranslate2 loads
these libraries with a plain LoadLibrary call rather than the newer
AddDllDirectory-aware search -- so `os.add_dll_directory()` (the normally
-recommended fix for this class of problem) does NOT work here; it has
to be the classic PATH environment variable.

Must be called before the first `import faster_whisper` / `import
ctranslate2` anywhere in the process, which is why every entry point
(pipeline.py, verify_setup.py, benchmark.py) calls this at module import
time, ahead of any faster_whisper import.
"""

from __future__ import annotations

import importlib.util
import os


def ensure_cuda_dlls_on_path() -> None:
    if os.name != "nt":
        return  # PATH-based DLL search is a Windows-only problem here

    for pkg_name in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        spec = importlib.util.find_spec(pkg_name)
        if spec is None or not spec.submodule_search_locations:
            continue  # package not installed -- GPU path just won't be available
        pkg_dir = list(spec.submodule_search_locations)[0]
        bin_dir = os.path.join(pkg_dir, "bin")
        if os.path.isdir(bin_dir) and bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
