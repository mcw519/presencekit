//! YuNet candidate decoding and greedy NMS, with a NumPy replay entry point.
#![forbid(unsafe_code)]

const STRIDES: [usize; 3] = [8, 16, 32];
const HEADS: usize = 12;
type Row = [f32; 15];

struct Candidate {
    row: Row,
    unstable_box: bool,
}

fn near_integer(value: f32, radius: f32) -> bool {
    let integer = value.round();
    radius > 0.0 && integer != 0.0 && (integer - value).abs() <= radius
}

fn unstable_dimension(origin: f32, size: f32, log_size: f32) -> bool {
    // NumPy's vector exp and expf can differ by a few ULPs. A difference in an
    // integer box can change NMS, even when the float output is within tolerance.
    if log_size == 0.0 {
        return false;
    }
    let uncertainty = size * (4.0 * f32::EPSILON);
    near_integer(size, uncertainty)
        || near_integer(origin, uncertainty * 0.5 + origin.abs() * f32::EPSILON)
}

fn decode(
    pad_width: usize,
    heads: &[&[f32]; HEADS],
    score_thr: f32,
    guarded: bool,
) -> Result<Vec<Candidate>, &'static str> {
    if pad_width == 0 || !pad_width.is_multiple_of(32) {
        return Err("invalid padded width");
    }
    let mut candidates = Vec::new();
    for (i, stride) in STRIDES.into_iter().enumerate() {
        let [cls, obj, bbox, kps] = [heads[i], heads[i + 3], heads[i + 6], heads[i + 9]];
        if obj.len() != cls.len()
            || bbox.len() / 4 != cls.len()
            || kps.len() / 10 != cls.len()
            || !bbox.len().is_multiple_of(4)
            || !kps.len().is_multiple_of(10)
        {
            return Err("inconsistent YuNet head sizes");
        }
        let cols = pad_width / stride;
        let step = stride as f32;
        for (idx, (&cls, &obj)) in cls.iter().zip(obj).enumerate() {
            let score = (cls.clamp(0.0, 1.0) * obj.clamp(0.0, 1.0)).sqrt();
            if score < score_thr || score.is_nan() {
                continue;
            }
            let c = (idx % cols) as f32;
            let r = (idx / cols) as f32;
            let raw = &bbox[idx * 4..(idx + 1) * 4];
            let (log_w, log_h) = (raw[2], raw[3]);
            let (w, h) = (log_w.exp() * step, log_h.exp() * step);
            let (x, y) = ((c + raw[0]) * step - w / 2.0, (r + raw[1]) * step - h / 2.0);
            let unstable_box = guarded
                && (unstable_dimension(x, w, log_w)
                    || unstable_dimension(y, h, log_h)
                    || [x, y, w, h]
                        .iter()
                        .any(|v| !(-2147483648.0..2147483648.0).contains(v)));
            let mut row = [0.0; 15];
            row[..4].copy_from_slice(&[x, y, w, h]);
            for (j, point) in kps[idx * 10..(idx + 1) * 10].iter().enumerate() {
                row[4 + j] = (point + if j % 2 == 0 { c } else { r }) * step;
            }
            row[14] = score;
            candidates.push(Candidate { row, unstable_box });
        }
    }
    Ok(candidates)
}

#[derive(Clone, Copy)]
struct Rect {
    x: i64,
    y: i64,
    w: i64,
    h: i64,
}

impl Rect {
    fn from_row(row: &Row) -> Self {
        let [x, y, w, h] = [row[0], row[1], row[2], row[3]].map(|v| i64::from(v as i32));
        Self { x, y, w, h }
    }

    fn overlap(self, other: Self) -> f32 {
        let total = self.w * self.h + other.w * other.h;
        if total == 0 {
            return 1.0;
        }
        let iw = (self.x + self.w).min(other.x + other.w) - self.x.max(other.x);
        let ih = (self.y + self.h).min(other.y + other.h) - self.y.max(other.y);
        let intersection = iw.max(0) * ih.max(0);
        (intersection as f64 / (total - intersection).max(1) as f64) as f32
    }
}

pub(crate) fn process(
    pad_width: usize,
    heads: &[&[f32]; HEADS],
    score_thr: f32,
    nms_thr: f32,
    top_k: usize,
    guarded: bool,
) -> Result<Option<Vec<Row>>, &'static str> {
    let mut candidates = decode(pad_width, heads, score_thr, guarded)?;
    if candidates.len() <= 1 {
        return Ok(Some(candidates.into_iter().map(|c| c.row).collect()));
    }
    // Production only applies the strict score filter when there is more than
    // one decoded row. Preserve insertion order for equal scores across strides.
    candidates.retain(|c| c.row[14] > score_thr);
    candidates.sort_by(|a, b| b.row[14].total_cmp(&a.row[14]));
    if top_k != 0 {
        candidates.truncate(top_k);
    }
    if candidates.len() > 1 && candidates.iter().any(|c| c.unstable_box) {
        return Ok(None);
    }
    let mut kept_boxes: Vec<Rect> = Vec::with_capacity(candidates.len());
    let mut rows = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let rect = Rect::from_row(&candidate.row);
        if kept_boxes
            .iter()
            .all(|other| rect.overlap(*other) <= nms_thr)
        {
            kept_boxes.push(rect);
            rows.push(candidate.row);
        }
    }
    Ok(Some(rows))
}

/// Boxed test adapter used by rust-anything record/diff.
pub fn postprocess(
    pad_width: i64,
    heads: Vec<Vec<f64>>,
    score_thr: f64,
    nms_thr: f64,
    top_k: i64,
) -> Result<Vec<Vec<f64>>, String> {
    let maps: [Vec<f32>; HEADS] = heads
        .into_iter()
        .map(|head| head.into_iter().map(|v| v as f32).collect())
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| "ValueError: expected twelve YuNet output heads")?;
    let refs = maps.each_ref().map(Vec::as_slice);
    let rows = process(
        usize::try_from(pad_width).unwrap_or_default(),
        &refs,
        score_thr as f32,
        nms_thr as f32,
        usize::try_from(top_k).unwrap_or_default(),
        false,
    )
    .map_err(|e| format!("ValueError: {e}"))?
    .ok_or("RuntimeError: unguarded decode requested NumPy")?;
    Ok(rows
        .into_iter()
        .map(|row| row.map(f64::from).to_vec())
        .collect())
}
