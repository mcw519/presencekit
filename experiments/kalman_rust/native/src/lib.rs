//! Python bindings for `oracle.py`: generated from the signatures in `port.rs` by rust-anything.
//! When refreshing, retain the registration of the experimental buffer bridge.

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

fn associate_inner(
    now: f64,
    gates: Vec<f64>,
    states: Bound<'_, PyAny>,
    tracks: Bound<'_, PyAny>,
    detections: Bound<'_, PyAny>,
) -> Result<Vec<Vec<f64>>, String> {
    let states: Vec<Vec<f64>> = crate::pyconv::FromPy::from_py(&states)?;
    let tracks: Vec<Vec<Vec<f64>>> = crate::pyconv::FromPy::from_py(&tracks)?;
    let detections: Vec<Vec<Vec<f64>>> = crate::pyconv::FromPy::from_py(&detections)?;
    let out = port::associate(now, gates, states, tracks, detections)?;
    Ok(out)
}

/// `associate`.
#[pyfunction]
#[pyo3(name = "associate", signature = (now, gates, states, tracks, detections))]
fn associate(
    py: Python<'_>,
    now: f64,
    gates: Vec<f64>,
    states: Bound<'_, PyAny>,
    tracks: Bound<'_, PyAny>,
    detections: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    associate_inner(now, gates, states, tracks, detections)
        .map_err(error)
        .and_then(|v| crate::pyconv::ToPy::to_py(&v, py))
}

#[pymodule]
#[pyo3(name = "_native")]
fn native_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(associate, m)?)?;
    buffer_bridge::register(m)
}
