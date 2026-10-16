"""Regression tests for src/cuda_dlls.py (the Windows CUDA PATH shim)."""

from __future__ import annotations

import importlib.util

import cuda_dlls


def test_noop_off_windows(monkeypatch):
    monkeypatch.setattr(cuda_dlls, "_is_windows", lambda: False)
    monkeypatch.setenv("PATH", "unchanged")
    cuda_dlls.ensure_cuda_dlls_on_path()
    assert cuda_dlls.os.environ["PATH"] == "unchanged"


def test_cpu_only_install_does_not_raise(monkeypatch):
    """A CPU-only Windows install has no `nvidia` package at all, and
    importlib.util.find_spec("nvidia.cublas") then raises
    ModuleNotFoundError rather than returning None. That must be treated as
    "GPU wheels not installed", not crash every pipeline import.
    """

    def _missing(name, *args, **kwargs):
        raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")

    monkeypatch.setattr(cuda_dlls, "_is_windows", lambda: True)
    monkeypatch.setenv("PATH", "unchanged")
    monkeypatch.setattr(importlib.util, "find_spec", _missing)
    cuda_dlls.ensure_cuda_dlls_on_path()
    assert cuda_dlls.os.environ["PATH"] == "unchanged"
