"""Minimal ONNX protobuf rewriting, without the ``onnx`` package (the runtime dependency stays onnxruntime only).

Many exported models (SFace among them) list every weight as a graph *input* as well as an initializer. From ONNX
IR v4 on, ONNX Runtime treats such initializers as overridable inputs, so it cannot constant-fold them: no
Conv+BatchNorm fusion, no pre-packed weights. :func:`strip_initializer_inputs` removes those duplicate inputs in
memory, which is exactly what ``onnxruntime/tools/python/remove_initializer_from_input.py`` does with ``onnx``.

Only the protobuf wire format is walked (tags, varints, length-delimited fields); nested messages other than the
graph are copied byte for byte.
"""

from __future__ import annotations

_MODEL_IR_VERSION, _MODEL_GRAPH = 1, 7  # onnx.ModelProto field numbers
_GRAPH_INITIALIZER, _GRAPH_INPUT = 5, 11  # onnx.GraphProto
_TENSOR_NAME = 8  # onnx.TensorProto
_VALUE_INFO_NAME = 1  # onnx.ValueInfoProto


def _varint(buf, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _encode_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _fields(buf, start: int = 0, end: int | None = None):
    """Yield ``(field_number, wire_type, field_start, value_start, value_end)`` for each field in ``buf[start:end]``."""
    end = len(buf) if end is None else end
    pos = start
    while pos < end:
        first = pos
        key, pos = _varint(buf, pos)
        num, wt = key >> 3, key & 7
        if wt == 0:
            vs = pos
            _, pos = _varint(buf, pos)
        elif wt == 1:
            vs, pos = pos, pos + 8
        elif wt == 2:
            n, vs = _varint(buf, pos)
            pos = vs + n
        elif wt == 5:
            vs, pos = pos, pos + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        if pos > end:
            raise ValueError("truncated protobuf field")
        yield num, wt, first, vs, pos


def _name(buf, start: int, end: int, field: int) -> str | None:
    for num, wt, _, vs, ve in _fields(buf, start, end):
        if num == field and wt == 2:
            return bytes(buf[vs:ve]).decode("utf-8")
    return None


def strip_initializer_inputs(model: bytes) -> tuple[bytes, int]:
    """Remove graph inputs that are also initializers. Returns ``(model_bytes, number_removed)``.

    The bytes are returned unchanged when nothing is removed or the model predates IR v4 (where initializers must
    be listed as inputs). Raises ``ValueError`` on malformed input.
    """
    buf = memoryview(model)
    ir = graph = None
    for num, wt, first, vs, ve in _fields(buf):
        if num == _MODEL_IR_VERSION and wt == 0:
            ir = _varint(buf, vs)[0]
        elif num == _MODEL_GRAPH and wt == 2:
            graph = (first, vs, ve)
    if graph is None:
        raise ValueError("not an ONNX model (no graph)")
    if ir is None or ir < 4:
        return model, 0
    first, gs, ge = graph
    inits = {
        _name(buf, vs, ve, _TENSOR_NAME) for num, wt, _, vs, ve in _fields(buf, gs, ge) if num == _GRAPH_INITIALIZER
    }
    inits.discard(None)
    keep, removed = [], 0
    for num, wt, f0, vs, ve in _fields(buf, gs, ge):
        if num == _GRAPH_INPUT and wt == 2 and _name(buf, vs, ve, _VALUE_INFO_NAME) in inits:
            removed += 1
            continue
        keep.append(buf[f0:ve])
    if not removed:
        return model, 0
    body = b"".join(keep)
    head = bytes(buf[:first]) + _encode_varint(_MODEL_GRAPH << 3 | 2) + _encode_varint(len(body))
    return head + body + bytes(buf[ge:]), removed
