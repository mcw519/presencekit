//! Copies the twelve ONNX output tensors into typed buffers in one PyO3 call.
#![forbid(unsafe_code)]

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::port::process;

#[pyfunction]
fn postprocess_buffer<'py>(
    py: Python<'py>,
    pad_width: usize,
    heads: [PyBuffer<f32>; 12],
    score_thr: f32,
    nms_thr: f32,
    top_k: i64,
) -> PyResult<Option<Bound<'py, PyBytes>>> {
    let buffers = heads
        .iter()
        .map(|head| head.to_vec(py))
        .collect::<PyResult<Vec<_>>>()?;
    let views = std::array::from_fn(|i| buffers[i].as_slice());
    let rows = process(
        pad_width,
        &views,
        score_thr,
        nms_thr,
        usize::try_from(top_k).unwrap_or_default(),
        true,
    )
    .map_err(PyValueError::new_err)?;
    let Some(rows) = rows else {
        return Ok(None);
    };
    PyBytes::new_with(py, rows.len() * 15 * size_of::<f32>(), |bytes| {
        for (chunk, value) in bytes
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(rows.into_iter().flatten())
        {
            *chunk = value.to_ne_bytes();
        }
        Ok(())
    })
    .map(Some)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(postprocess_buffer, module)?)
}
