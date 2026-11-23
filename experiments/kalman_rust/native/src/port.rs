//! One-frame Kalman association for the PresenceKit experiment.
#![forbid(unsafe_code)]

const FORBIDDEN: f64 = 1e6;
const DECISION_MARGIN: f64 = 1e-5;
const ACCELERATION_VARIANCE: [f64; 4] = [0.25, 0.25, 0.1 * 0.1, 0.1 * 0.1];
const MEASUREMENT_VARIANCE: [f64; 4] = [0.01 * 0.01, 0.01 * 0.01, 0.015 * 0.015, 0.015 * 0.015];
type Bbox = [f64; 4];

struct Filter {
    state: Vec<f64>,
    predicted: Bbox,
}

impl Filter {
    fn predict(mut state: Vec<f64>, now: f64) -> Self {
        let dt = now - state[24];
        if dt > 0.0 {
            for (axis, variance) in ACCELERATION_VARIANCE.into_iter().enumerate() {
                state[axis] += dt * state[axis + 4];
                let k = 8 + 4 * axis;
                let (p00, p01, p10, p11) = (state[k], state[k + 1], state[k + 2], state[k + 3]);
                let cross_noise = variance * dt.powi(3) / 2.0;
                state[k] = (p00 + dt * p10) + (p01 + dt * p11) * dt + variance * dt.powi(4) / 4.0;
                state[k + 1] = p01 + dt * p11 + cross_noise;
                state[k + 2] = p10 + dt * p11 + cross_noise;
                state[k + 3] = p11 + variance * dt.powi(2);
            }
            state[2] = state[2].max(1e-3);
            state[3] = state[3].max(1e-3);
            state[24] = now;
        }
        let cx = state[0];
        let cy = state[1];
        let hw = state[2].max(1e-4) / 2.0;
        let hh = state[3].max(1e-4) / 2.0;
        Self {
            state,
            predicted: [cx - hw, cy - hh, cx + hw, cy + hh],
        }
    }

    fn center_distance(&self, measurement: Bbox) -> f64 {
        let dx = measurement[0] - self.state[0];
        let dy = measurement[1] - self.state[1];
        dx * (dx / (self.state[8] + MEASUREMENT_VARIANCE[0]))
            + dy * (dy / (self.state[12] + MEASUREMENT_VARIANCE[1]))
    }

    fn update(&mut self, measurement: Bbox) {
        // The transition and noise models preserve four independent position/velocity blocks.
        for (axis, (z, variance)) in measurement
            .into_iter()
            .zip(MEASUREMENT_VARIANCE)
            .enumerate()
        {
            let k = 8 + 4 * axis;
            let p00 = self.state[k];
            let p01 = self.state[k + 1];
            let p10 = self.state[k + 2];
            let p11 = self.state[k + 3];
            let inverse = 1.0 / (p00 + variance);
            let kp = p00 * inverse;
            let kv = p10 * inverse;
            let innovation = z - self.state[axis];
            self.state[axis] += kp * innovation;
            self.state[axis + 4] += kv * innovation;
            self.state[k] = (1.0 - kp) * p00;
            self.state[k + 1] = (1.0 - kp) * p01;
            self.state[k + 2] = -kv * p00 + p10;
            self.state[k + 3] = -kv * p01 + p11;
        }
    }
}

pub(crate) struct Observation<'a, E> {
    pub bbox: Bbox,
    pub body: Option<Bbox>,
    pub embedding: &'a [E],
    pub appearance: &'a [f64],
}

impl<'a> Observation<'a, f64> {
    fn decode(row: &'a [Vec<f64>]) -> Result<Self, String> {
        let [bbox, body, embedding, appearance] = row else {
            return Err("ValueError: invalid observation".into());
        };
        let bbox: Bbox = bbox
            .as_slice()
            .try_into()
            .map_err(|_| "ValueError: invalid observation")?;
        let body = if body.is_empty() {
            None
        } else {
            Some(
                body.as_slice()
                    .try_into()
                    .map_err(|_| "ValueError: invalid observation")?,
            )
        };
        Ok(Self {
            bbox,
            body,
            embedding,
            appearance,
        })
    }
}

fn measurement([x0, y0, x1, y1]: Bbox) -> Bbox {
    [(x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0]
}

fn iou(a: &Bbox, b: &Bbox) -> f64 {
    let width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = width * height;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

fn pair_cost(
    det: &Observation<'_, impl Copy + Into<f64>>,
    track: &Observation<'_, impl Copy + Into<f64>>,
    filter: &Filter,
    z: Bbox,
    &[motion, iou_gate, strong, weak, appearance_gate]: &[f64; 5],
    guarded: bool,
) -> Result<PairCost, String> {
    let mut overlap = iou(&det.bbox, &filter.predicted);
    if let (Some(det_body), Some(track_body)) = (det.body, track.body) {
        overlap = overlap.max(iou(&det_body, &track_body));
    }
    let distance = filter.center_distance(z);
    let cosine = if det.embedding.is_empty() || track.embedding.is_empty() {
        None
    } else {
        if det.embedding.len() != track.embedding.len() {
            return Err("ValueError: incompatible embedding lengths".into());
        }
        let sum = det
            .embedding
            .iter()
            .zip(track.embedding)
            .map(|(&a, &b)| a.into() * b.into())
            .sum::<f64>();
        Some(f64::from(sum as f32))
    };
    if guarded && cosine.is_some_and(|v| near(v, weak) || near(v, strong)) {
        return Ok(PairCost::Reference);
    }
    if distance > motion && overlap < iou_gate && !cosine.is_some_and(|v| v >= strong)
        || cosine.is_some_and(|v| v < weak)
    {
        return Ok(PairCost::Forbidden);
    }
    let appearance = (!det.appearance.is_empty() && det.appearance.len() == track.appearance.len())
        .then(|| {
            det.appearance
                .iter()
                .zip(track.appearance)
                .map(|(a, b)| (a.max(0.0) * b.max(0.0)).sqrt())
                .sum::<f64>()
        });
    if cosine.is_none() && appearance.is_some_and(|v| v < appearance_gate) {
        return Ok(PairCost::Forbidden);
    }
    let mut score = overlap + (-distance / motion).exp();
    let mut weight = 2.0;
    if let Some(value) = cosine {
        score += 2.0 * (value.max(0.0) / strong.max(1e-6)).min(1.0);
        weight += 2.0;
    }
    if let Some(value) = appearance {
        score += 1.5 * ((value - 0.5) / 0.5).clamp(0.0, 1.0);
        weight += 1.5;
    }
    let score = score / weight;
    if guarded && near(score, 0.3) {
        return Ok(PairCost::Reference);
    }
    if score >= 0.3 {
        Ok(PairCost::Eligible(1.0 - score, cosine.unwrap_or(0.0)))
    } else {
        Ok(PairCost::Forbidden)
    }
}

enum PairCost {
    Eligible(f64, f64),
    Forbidden,
    Reference,
}

fn near(value: f64, threshold: f64) -> bool {
    (value - threshold).abs() <= DECISION_MARGIN
}

fn hungarian(cost: &[Vec<(f64, f64)>], cols: usize, guarded: bool) -> Option<Vec<(usize, usize)>> {
    let rows = cost.len();
    if rows == 0 || cols == 0 {
        return Some(Vec::new());
    }
    let transpose = rows > cols;
    let (n, m) = (rows.min(cols), rows.max(cols));
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut p = vec![0; m + 1];
    let mut way = vec![0; m + 1];
    let mut used = vec![false; m + 1];
    let mut best = vec![f64::INFINITY; m + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        used.fill(false);
        best.fill(f64::INFINITY);
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=m {
                if used[j] {
                    continue;
                }
                let value = if transpose {
                    cost[j - 1][i0 - 1].0
                } else {
                    cost[i0 - 1][j - 1].0
                };
                let reduced = value - u[i0] - v[j];
                if reduced < best[j] {
                    best[j] = reduced;
                    way[j] = j0;
                }
                if best[j] < delta {
                    delta = best[j];
                    j1 = j;
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    best[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    if guarded {
        // An eligible unassigned edge with nearly zero reduced cost can be part of an
        // equally good assignment. Use the reference backend's tie-breaking and numerics.
        for (j, &assigned) in p.iter().enumerate().skip(1) {
            for (i, &potential) in u.iter().enumerate().skip(1) {
                let value = if transpose {
                    cost[j - 1][i - 1].0
                } else {
                    cost[i - 1][j - 1].0
                };
                if i != assigned && value < FORBIDDEN && near(value - potential - v[j], 0.0) {
                    return None;
                }
            }
        }
    }
    let mut pairs = Vec::new();
    for (j, &i) in p.iter().enumerate().skip(1) {
        if i != 0 {
            let pair = if transpose {
                (j - 1, i - 1)
            } else {
                (i - 1, j - 1)
            };
            if cost[pair.0][pair.1].0 < FORBIDDEN {
                pairs.push(pair);
            }
        }
    }
    pairs.sort_unstable();
    Some(pairs)
}

pub fn associate(
    now: f64,
    gates: Vec<f64>,
    states: Vec<Vec<f64>>,
    tracks: Vec<Vec<Vec<f64>>>,
    detections: Vec<Vec<Vec<f64>>>,
) -> Result<Vec<Vec<f64>>, String> {
    let tracks = tracks
        .iter()
        .map(|row| Observation::decode(row))
        .collect::<Result<Vec<_>, _>>()?;
    let dets = detections
        .iter()
        .map(|row| Observation::decode(row))
        .collect::<Result<Vec<_>, _>>()?;
    associate_frame(now, gates, states, &tracks, &dets, false)?
        .ok_or_else(|| "RuntimeError: unguarded association requested a reference".into())
}

pub(crate) fn associate_frame<E: Copy + Into<f64>>(
    now: f64,
    gates: Vec<f64>,
    states: Vec<Vec<f64>>,
    tracks: &[Observation<'_, E>],
    dets: &[Observation<'_, E>],
    guarded: bool,
) -> Result<Option<Vec<Vec<f64>>>, String> {
    let gates: [f64; 5] = gates
        .try_into()
        .map_err(|_| "ValueError: invalid Kalman state or gates")?;
    if states.len() != tracks.len() || states.iter().any(|r| r.len() != 25) {
        return Err("ValueError: invalid Kalman state or gates".into());
    }
    let mut filters: Vec<_> = states
        .into_iter()
        .map(|state| Filter::predict(state, now))
        .collect();
    let measures: Vec<_> = dets.iter().map(|det| measurement(det.bbox)).collect();
    let mut costs = Vec::with_capacity(dets.len());
    for (det, &z) in dets.iter().zip(&measures) {
        let mut row = Vec::with_capacity(tracks.len());
        for (track, filter) in tracks.iter().zip(&filters) {
            match pair_cost(det, track, filter, z, &gates, guarded)? {
                PairCost::Eligible(cost, similarity) => row.push((cost, similarity)),
                PairCost::Forbidden => row.push((FORBIDDEN, 0.0)),
                PairCost::Reference => return Ok(None),
            }
        }
        costs.push(row);
    }
    let Some(matches) = hungarian(&costs, tracks.len(), guarded) else {
        return Ok(None);
    };
    for &(i, j) in &matches {
        filters[j].update(measures[i]);
    }
    let mut rows = Vec::with_capacity(tracks.len() + matches.len());
    for mut filter in filters {
        filter.state.extend(filter.predicted);
        rows.push(filter.state);
    }
    rows.extend(
        matches
            .into_iter()
            .map(|(i, j)| vec![i as f64, j as f64, costs[i][j].1]),
    );
    Ok(Some(rows))
}
