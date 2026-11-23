//! Python bindings for `oracle.py`: generated from the signatures in `port.rs` by rust-anything.
//! Regenerate with `rust-anything refresh` after changing them.

use pyo3::prelude::*;

mod buffer_bridge;
pub mod js;
pub mod port;
pub mod pyconv;
pub mod pynum;

#[allow(unused_imports)]
use pynum::Num;

fn error(message: String) -> PyErr {
    use pyo3::exceptions as e;
    let (kind, text) = message.split_once(": ").unwrap_or(("", message.as_str()));
    let text = text.to_string();
    match kind {
        "TypeError" => e::PyTypeError::new_err(text),
        "KeyError" => e::PyKeyError::new_err(text),
        "IndexError" => e::PyIndexError::new_err(text),
        "OverflowError" => e::PyOverflowError::new_err(text),
        "ZeroDivisionError" => e::PyZeroDivisionError::new_err(text),
        "AttributeError" => e::PyAttributeError::new_err(text),
        "RuntimeError" => e::PyRuntimeError::new_err(text),
        "NotImplementedError" => e::PyNotImplementedError::new_err(text),
        "LookupError" => e::PyLookupError::new_err(text),
        "ArithmeticError" => e::PyArithmeticError::new_err(text),
        "UnicodeError" => e::PyUnicodeError::new_err(text),
        "ValueError" => e::PyValueError::new_err(text),
        _ => e::PyValueError::new_err(message),
    }
}

fn postprocess_inner(
    pad_width: i64,
    heads: Bound<'_, PyAny>,
    score_thr: f64,
    nms_thr: f64,
    top_k: i64,
) -> Result<Vec<Vec<f64>>, String> {
    let heads: Vec<Vec<f64>> = crate::pyconv::FromPy::from_py(&heads)?;
    let out = port::postprocess(pad_width, heads, score_thr, nms_thr, top_k)?;
    Ok(out)
}

/// `postprocess`.
#[pyfunction]
#[pyo3(name = "postprocess", signature = (pad_width, heads, score_thr, nms_thr, top_k))]
fn postprocess(
    py: Python<'_>,
    pad_width: i64,
    heads: Bound<'_, PyAny>,
    score_thr: f64,
    nms_thr: f64,
    top_k: i64,
) -> PyResult<Py<PyAny>> {
    postprocess_inner(pad_width, heads, score_thr, nms_thr, top_k)
        .map_err(error)
        .and_then(|v| crate::pyconv::ToPy::to_py(&v, py))
}

#[pymodule]
#[pyo3(name = "_native")]
fn native_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(postprocess, m)?)?;
    buffer_bridge::register(m)?;
    Ok(())
}
