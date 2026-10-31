"""Execution-provider selection for ONNX Runtime.

Providers are named by their ORT name (``CUDAExecutionProvider``) or a short alias (``cuda``). ``auto`` takes the
first available GPU provider (CUDA, ROCm, DirectML) and otherwise the CPU. NPU-style providers (CoreML, NNAPI, QNN,
OpenVINO) are used only when named: operator coverage differs per model, and a partitioned graph can be slower than
the CPU (SFace on CoreML ran about 3x slower than on the CPU on an M-series Mac). The CPU provider always ends the
list, so every model keeps a working fallback.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

CPU = "CPUExecutionProvider"
ALIASES = {
    "cpu": CPU,
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "nnapi": "NnapiExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "xnnpack": "XnnpackExecutionProvider",
    "webgpu": "WebGpuExecutionProvider",
}
AUTO_ORDER = ("CUDAExecutionProvider", "ROCMExecutionProvider", "DmlExecutionProvider")


def canonical(name: str) -> str:
    """ORT provider name for an alias (``coreml`` -> ``CoreMLExecutionProvider``); other names pass through."""
    n = name.strip()
    return ALIASES.get(n.lower(), n)


def available() -> list[str]:
    """Providers compiled into the installed onnxruntime package."""
    import onnxruntime as ort

    return list(ort.get_available_providers())


def parse(spec: str | Iterable[str]) -> tuple[str, ...]:
    """``"coreml, cpu"`` or ``["coreml", "cpu"]`` -> ``("coreml", "cpu")``."""
    items = spec.split(",") if isinstance(spec, str) else spec
    return tuple(s.strip() for s in items if s and s.strip())


def resolve(
    requested: str | Iterable[str],
    provider_options: Mapping[str, Mapping[str, object]] | None = None,
    avail: Iterable[str] | None = None,
) -> tuple[list[str], list[dict], list[str]]:
    """Requested providers -> ``(providers, options, notes)`` for ``InferenceSession``.

    Unavailable providers are skipped with a note; duplicates are dropped; CPU is always last. ``provider_options``
    is keyed by alias or ORT name.
    """
    have = list(available() if avail is None else avail)
    opts = {canonical(k): dict(v) for k, v in (provider_options or {}).items()}
    names: list[str] = []
    notes: list[str] = []
    for item in parse(requested) or ("cpu",):
        if item.lower() == "auto":
            cand = [p for p in AUTO_ORDER if p in have]
        else:
            p = canonical(item)
            if p not in have:
                notes.append(f"{p} not available in this onnxruntime build")
                continue
            cand = [p]
        names += [p for p in cand if p not in names and p != CPU]
    names.append(CPU)
    return names, [opts.get(p, {}) for p in names], notes


def short(provider: str) -> str:
    """``CoreMLExecutionProvider`` -> ``coreml`` (the alias when there is one)."""
    for alias, full in ALIASES.items():
        if full == provider and alias != "dml":
            return alias
    return provider.removesuffix("ExecutionProvider").lower()
