"""ONNX Runtime sessions: one options object, a process-wide session cache and a small model wrapper.

:class:`OrtOptions` holds everything that decides how a model runs (execution providers, threads, graph
optimisation). :func:`load_model` returns an :class:`OrtModel`; sessions are cached per (file, options), so every
pipeline and connection in the process shares one session per model (``InferenceSession.run`` is thread-safe).
Duplicate initializer inputs are stripped in memory before loading (:mod:`.onnx_graph`); model files are only read.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import providers as _prov
from .onnx_graph import strip_initializer_inputs

log = logging.getLogger(__name__)
GRAPH_LEVELS = ("disable", "basic", "extended", "all")


@dataclass(frozen=True)
class OrtOptions:
    """How ONNX models are run. Immutable; derive variants with :meth:`replace`.

    ``providers``: aliases or ORT names, or ``("auto",)``; CPU is always appended as the fallback.
    ``intra_op_threads``: 0 lets ORT choose (one per physical core). ``sanitize``: strip duplicate initializer
    inputs in memory so ORT can fold constants (see :mod:`presence_demo.inference.onnx_graph`).
    """

    providers: tuple[str, ...] = ("cpu",)
    provider_options: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    intra_op_threads: int = 0
    inter_op_threads: int = 1
    graph_optimization: str = "all"
    sanitize: bool = True
    log_severity: int = 3  # 0 verbose .. 4 fatal; 3 = errors only

    def __post_init__(self):
        object.__setattr__(self, "providers", _prov.parse(self.providers) or ("cpu",))
        if self.graph_optimization not in GRAPH_LEVELS:
            raise ValueError(f"graph_optimization must be one of {GRAPH_LEVELS}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides) -> OrtOptions:
        """Defaults overridden by ``PRESENCE_ORT_PROVIDERS`` (e.g. ``coreml,cpu``), ``PRESENCE_ORT_THREADS`` and
        ``PRESENCE_ORT_GRAPH_OPT``, then by keyword arguments."""
        env = os.environ if env is None else env
        kw: dict = {}
        if env.get("PRESENCE_ORT_PROVIDERS"):
            kw["providers"] = _prov.parse(env["PRESENCE_ORT_PROVIDERS"])
        if env.get("PRESENCE_ORT_THREADS"):
            kw["intra_op_threads"] = int(env["PRESENCE_ORT_THREADS"])
        if env.get("PRESENCE_ORT_GRAPH_OPT"):
            kw["graph_optimization"] = env["PRESENCE_ORT_GRAPH_OPT"]
        return cls(**{**kw, **overrides})

    def replace(self, **kw) -> OrtOptions:
        return replace(self, **kw)

    def key(self) -> tuple:
        po = tuple(
            sorted(
                (_prov.canonical(k), tuple(sorted((str(a), str(b)) for a, b in v.items())))
                for k, v in self.provider_options.items()
            )
        )
        return (
            self.providers,
            po,
            self.intra_op_threads,
            self.inter_op_threads,
            self.graph_optimization,
            self.sanitize,
            self.log_severity,
        )

    def __hash__(self):
        return hash(self.key())

    def session_options(self):
        import onnxruntime as ort

        levels = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        so = ort.SessionOptions()
        so.graph_optimization_level = levels[self.graph_optimization]
        so.intra_op_num_threads = self.intra_op_threads
        so.inter_op_num_threads = self.inter_op_threads
        so.log_severity_level = self.log_severity
        return so


class OrtModel:
    """A loaded ONNX model: the (shared) session, its I/O names and the providers actually in use."""

    def __init__(
        self,
        session,
        path: Path | None,
        options: OrtOptions,
        notes: Sequence[str] = (),
        load_ms: float = 0.0,
        sanitized: int = 0,
    ):
        self.session, self.path, self.options = session, path, options
        self.notes, self.load_ms, self.sanitized = list(notes), load_ms, sanitized
        self.input_names = [i.name for i in session.get_inputs()]
        self.output_names = [o.name for o in session.get_outputs()]

    @property
    def providers(self) -> list[str]:
        return list(self.session.get_providers())

    @property
    def accelerator(self) -> str:
        """Main provider, e.g. ``cpu (onnxruntime)`` or ``coreml (onnxruntime)``."""
        return f"{_prov.short(self.providers[0])} (onnxruntime)"

    def run(self, feeds: Mapping[str, np.ndarray], outputs: Sequence[str] | None = None) -> list[np.ndarray]:
        return self.session.run(list(outputs) if outputs else None, dict(feeds))

    def info(self) -> dict:
        return {
            "model": self.path.name if self.path else "",
            "providers": [_prov.short(p) for p in self.providers],
            "requested": list(self.options.providers),
            "threads": self.options.intra_op_threads,
            "graph_optimization": self.options.graph_optimization,
            "sanitized_inputs": self.sanitized,
            "load_ms": round(self.load_ms, 1),
            "notes": list(self.notes),
        }


_CACHE: dict[tuple, OrtModel] = {}
_LOCK = threading.Lock()


def load_model(path: str | Path, options: OrtOptions | None = None, *, cache: bool = True) -> OrtModel:
    """Load an ONNX file (shared per process for equal options). Falls back to the CPU provider when the requested
    accelerator fails to create the session."""
    import onnxruntime as ort

    opts = options or OrtOptions()
    p = Path(path).resolve()
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns, opts.key())
    with _LOCK:  # also keeps two callers from building the same session twice
        if cache and key in _CACHE:
            return _CACHE[key]
        t0 = time.perf_counter()
        names, prov_opts, notes = _prov.resolve(opts.providers, opts.provider_options)
        source: str | bytes = str(p)
        removed = 0
        if opts.sanitize:
            try:
                data, removed = strip_initializer_inputs(p.read_bytes())
                if removed:
                    source = data
            except ValueError as e:
                notes.append(f"graph not sanitized ({e})")
        so = opts.session_options()
        try:
            sess = ort.InferenceSession(source, so, providers=names, provider_options=prov_opts)
        except Exception as e:  # noqa: BLE001 - missing GPU libraries, unsupported graph on the accelerator...
            if names == [_prov.CPU]:
                raise
            notes.append(f"{names[0]} failed ({type(e).__name__}: {e}); using CPU")
            log.warning("onnxruntime: %s failed for %s, using CPU: %s", names[0], p.name, e)
            sess = ort.InferenceSession(source, so, providers=[_prov.CPU])
        model = OrtModel(sess, p, opts, notes, (time.perf_counter() - t0) * 1000, removed)
        if cache:
            _CACHE[key] = model
        return model


def clear_cache() -> None:
    """Drop the shared sessions (tests; or to reload models after replacing files)."""
    with _LOCK:
        _CACHE.clear()


def loaded_models() -> list[dict]:
    """``OrtModel.info()`` of every cached session (diagnostics)."""
    with _LOCK:
        return [m.info() for m in _CACHE.values()]
