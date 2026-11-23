//! Contiguous feature buffers without per-element Python conversion or a newer Python ABI.
#![forbid(unsafe_code)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::port::{Observation, associate_frame};

type Bbox = [f64; 4];
type PyObservation<'py> = (
    Bbox,
    Option<Bbox>,
    Option<Bound<'py, PyBytes>>,
    Option<Bound<'py, PyBytes>>,
);

struct Input {
    bbox: Bbox,
    body: Option<Bbox>,
    embedding: Vec<f32>,
    appearance: Vec<f64>,
}

impl Input {
    fn from_buffer((bbox, body, embedding, appearance): PyObservation<'_>) -> PyResult<Self> {
        Ok(Self {
            bbox,
            body,
            embedding: decode_buffer(embedding, f32::from_ne_bytes)?,
            appearance: decode_buffer(appearance, f64::from_ne_bytes)?,
        })
    }

    fn as_observation(&self) -> Observation<'_, f32> {
        Observation {
            bbox: self.bbox,
            body: self.body,
            embedding: &self.embedding,
            appearance: &self.appearance,
        }
    }
}

fn decode_buffer<T, const N: usize>(
    input: Option<Bound<'_, PyBytes>>,
    decode: fn([u8; N]) -> T,
) -> PyResult<Vec<T>> {
    let Some(buffer) = input else {
        return Ok(Vec::new());
    };
    let (chunks, rest) = buffer.as_bytes().as_chunks::<N>();
    if !rest.is_empty() {
        return Err(PyValueError::new_err(
            "feature buffer has an incomplete element",
        ));
    }
    Ok(chunks.iter().copied().map(decode).collect())
}

#[pyfunction]
#[pyo3(signature = (now, gates, states, tracks, detections))]
fn associate_buffer(
    now: f64,
    gates: Vec<f64>,
    states: Vec<Vec<f64>>,
    tracks: Vec<PyObservation<'_>>,
    detections: Vec<PyObservation<'_>>,
) -> PyResult<Option<Vec<Vec<f64>>>> {
    let tracks = tracks
        .into_iter()
        .map(Input::from_buffer)
        .collect::<PyResult<Vec<_>>>()?;
    let detections = detections
        .into_iter()
        .map(Input::from_buffer)
        .collect::<PyResult<Vec<_>>>()?;
    let tracks = tracks.iter().map(Input::as_observation).collect::<Vec<_>>();
    let detections = detections
        .iter()
        .map(Input::as_observation)
        .collect::<Vec<_>>();
    associate_frame(now, gates, states, &tracks, &detections, true).map_err(crate::error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(associate_buffer, module)?)
}
