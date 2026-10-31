"""Inference layer (v0.14): every neural model runs on ONNX Runtime through this package.

* :class:`OrtOptions` - execution providers, threads, graph optimisation (``OrtOptions.from_env()`` reads
  ``PRESENCE_ORT_PROVIDERS`` / ``PRESENCE_ORT_THREADS`` / ``PRESENCE_ORT_GRAPH_OPT``).
* :func:`load_model` -> :class:`OrtModel` - one shared session per (file, options) in the process.
* :data:`CATALOG` / :class:`ModelStore` - which model files exist, where they live, download and sha256 checks.

Depends on numpy and onnxruntime only (no OpenCV, no other presence_demo layer), so it can be reused on its own.
"""

from .models import CATALOG, ModelSpec, ModelStore, default_root, default_store
from .onnx_graph import strip_initializer_inputs
from .providers import available as available_providers
from .providers import resolve as resolve_providers
from .session import OrtModel, OrtOptions, clear_cache, load_model, loaded_models

__all__ = [
    "CATALOG",
    "ModelSpec",
    "ModelStore",
    "OrtModel",
    "OrtOptions",
    "available_providers",
    "clear_cache",
    "default_root",
    "default_store",
    "load_model",
    "loaded_models",
    "resolve_providers",
    "strip_initializer_inputs",
]
