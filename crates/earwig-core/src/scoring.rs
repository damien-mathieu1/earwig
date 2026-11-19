//! Hybrid scoring — the heart of the recommender.
//!
//! ```text
//! score = ( w_audio · cos(audio_src, audio_cand)
//!         + w_text  · cos(text_src,  text_cand)
//!         + w_bpm   · bpm_score )  /  (w_audio + w_text + w_bpm)
//!
//! bpm_delta = min(|a − b|, |2a − b|, |a − 2b|)      // half/double-time aware
//! bpm_score = max(0, 1 − bpm_delta / bpm_tolerance)
//! ```
//!
//! Embeddings are L2-normalised at ingestion, so cosine similarity is a plain dot
//! product (SIMD via `simsimd`).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Weights {
    pub w_audio: f32,
    pub w_text: f32,
    pub w_bpm: f32,
    pub bpm_tolerance: f32,
}

impl Default for Weights {
    fn default() -> Self {
        // tuned with `earwig eval --grid` (MTG-Jamendo genre/mood + a tagged library): BPM only hurts relevance
        Self {
            w_audio: 0.7,
            w_text: 0.3,
            w_bpm: 0.0,
            bpm_tolerance: 20.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SubScores {
    pub score: f32,
    pub audio: f32,
    pub text: f32,
    pub bpm_delta: f32,
    pub bpm_score: f32,
}

/// What the scorer needs from a track.
pub trait Scorable {
    fn audio(&self) -> &[f32];
    fn text(&self) -> &[f32];
    fn bpm(&self) -> f32;
}

/// Cosine similarity of two L2-normalised vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    f32::dot(a, b).unwrap_or(0.0) as f32
}

pub fn bpm_delta(a: f32, b: f32) -> f32 {
    (a - b)
        .abs()
        .min((2.0 * a - b).abs())
        .min((a - 2.0 * b).abs())
}

pub fn hybrid<Q: Scorable, T: Scorable>(src: &Q, cand: &T, w: &Weights) -> SubScores {
    let audio = cosine(src.audio(), cand.audio());
    let text = cosine(src.text(), cand.text());
    let bpm_delta = bpm_delta(src.bpm(), cand.bpm());
    let bpm_score = (1.0 - bpm_delta / w.bpm_tolerance.max(1e-3)).max(0.0);
    let score = combine(w, audio, text, bpm_score);
    SubScores {
        score,
        audio,
        text,
        bpm_delta,
        bpm_score,
    }
}

/// The weighted mean itself (reused by `earwig eval` to re-weight cached sub-scores).
pub fn combine(w: &Weights, audio: f32, text: f32, bpm_score: f32) -> f32 {
    let (wa, wt, wb) = (w.w_audio.max(0.0), w.w_text.max(0.0), w.w_bpm.max(0.0));
    let sum = wa + wt + wb;
    if sum > 0.0 {
        (wa * audio + wt * text + wb * bpm_score) / sum
    } else {
        0.0
    }
}

/// Exact top-k over the in-RAM catalogue. `query` is either a catalogue track (pass its
/// index as `exclude`) or an external file analysed on the fly.
///
/// The scan only computes the components that have a weight, runs on all cores once the
/// catalogue is large, and fills the full sub-scores for the k winners only.
// shortcut: O(n) per query (~4 ms at 100k tracks, ~40 ms at 1M on 10 cores); add an HNSW candidate stage past ~1M
pub fn top_k<Q: Scorable + Sync, T: Scorable + Sync>(
    tracks: &[T],
    query: &Q,
    exclude: Option<usize>,
    w: &Weights,
    k: usize,
) -> Vec<(usize, SubScores)> {
    let tol = w.bpm_tolerance.max(1e-3);
    let mut scored: Vec<(usize, f32)> = tracks
        .par_iter()
        .with_min_len(4096) // small libraries stay on one thread
        .enumerate()
        .filter(|&(i, _)| Some(i) != exclude)
        .map(|(i, t)| {
            let audio = if w.w_audio > 0.0 {
                cosine(query.audio(), t.audio())
            } else {
                0.0
            };
            let text = if w.w_text > 0.0 {
                cosine(query.text(), t.text())
            } else {
                0.0
            };
            let bpm = if w.w_bpm > 0.0 {
                (1.0 - bpm_delta(query.bpm(), t.bpm()) / tol).max(0.0)
            } else {
                0.0
            };
            (i, combine(w, audio, text, bpm))
        })
        .collect();
    let k = k.min(scored.len());
    let by_score = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1);
    if k < scored.len() {
        scored.select_nth_unstable_by(k, by_score);
        scored.truncate(k);
    }
    scored.sort_unstable_by(by_score);
    scored
        .into_iter()
        .map(|(i, _)| (i, hybrid(query, &tracks[i], w)))
        .collect()
}

/// Weighted mix of per-track component scores (`(weight, one score per track)`), for queries
/// that combine a reference track, moods, styles, BPM… Each component is standardised (z-score)
/// over the library first, so scores on different scales mix fairly: audio↔audio cosines sit
/// around 0.8, text↔audio cosines around 0.2. Components with no weight or no spread are ignored.
/// `exclude(i)` removes track `i` from the results (the seeds, already-heard tracks…) without
/// changing the standardisation, which is computed over the whole library.
pub fn mix_top_k(
    components: &[(f32, Vec<f32>)],
    exclude: impl Fn(usize) -> bool,
    k: usize,
) -> Vec<(usize, f32)> {
    mix_top_k_by(components, exclude, k, |_| true)
}

/// [`mix_top_k`] where `accept(i)` is asked about each candidate in rank order and may refuse
/// it, e.g. to cap the tracks per artist. A refused track does not count towards `k`.
pub fn mix_top_k_by(
    components: &[(f32, Vec<f32>)],
    exclude: impl Fn(usize) -> bool,
    k: usize,
    mut accept: impl FnMut(usize) -> bool,
) -> Vec<(usize, f32)> {
    let n = components.iter().map(|(_, v)| v.len()).max().unwrap_or(0);
    let mut total = vec![0f32; n];
    let mut weights = 0.0;
    for (w, v) in components.iter().filter(|(w, v)| *w > 0.0 && v.len() == n) {
        let mean = v.iter().sum::<f32>() / n as f32;
        let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32).sqrt();
        if std > 1e-6 {
            total
                .iter_mut()
                .zip(v)
                .for_each(|(t, x)| *t += w * (x - mean) / std);
            weights += w;
        }
    }
    let mut pool: Vec<(usize, f32)> = (0..n)
        .filter(|&i| !exclude(i))
        .map(|i| {
            (
                i,
                if weights > 0.0 {
                    total[i] / weights
                } else {
                    0.0
                },
            )
        })
        .collect();
    let by_score = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1);
    // rank the best few first; widen only when `accept` refuses too many
    let mut out = Vec::with_capacity(k);
    let mut take = k.max(1) * 4;
    while out.len() < k && !pool.is_empty() {
        let rest = if take < pool.len() {
            pool.select_nth_unstable_by(take, by_score);
            pool.split_off(take)
        } else {
            Vec::new()
        };
        pool.sort_unstable_by(by_score);
        out.extend(pool.iter().filter(|&&(i, _)| accept(i)).take(k - out.len()));
        pool = rest;
        take *= 8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct T(Vec<f32>, Vec<f32>, f32);
    impl Scorable for T {
        fn audio(&self) -> &[f32] {
            &self.0
        }
        fn text(&self) -> &[f32] {
            &self.1
        }
        fn bpm(&self) -> f32 {
            self.2
        }
    }
    const W: Weights = Weights {
        w_audio: 1.0,
        w_text: 1.0,
        w_bpm: 1.0,
        bpm_tolerance: 20.0,
    };

    #[test]
    fn mix_standardises_components_before_weighting() {
        // component a: tiny spread, favours track 2; component b: big spread, favours track 1
        let a = vec![0.20, 0.20, 0.22, 0.20];
        let b = vec![0.1, 0.9, 0.5, 0.1];
        let top = |wa, wb| mix_top_k(&[(wa, a.clone()), (wb, b.clone())], |i| i == 0, 1)[0].0;
        assert_eq!(top(1.0, 0.0), 2);
        assert_eq!(top(0.0, 1.0), 1);
        // equal weights: a's small raw gap counts as much as b's big one once standardised
        assert_eq!(top(1.0, 1.0), 2);
        assert!(mix_top_k(&[], |_| false, 3).is_empty());
        // excluding the best ones leaves the next ones, scores unchanged
        let all = mix_top_k(&[(1.0, b.clone())], |_| false, 4);
        let rest = mix_top_k(&[(1.0, b.clone())], |i| i == 1 || i == 2, 4);
        assert_eq!(rest.iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 3]);
        assert_eq!(rest[0].1, all.iter().find(|x| x.0 == 0).unwrap().1);
    }

    #[test]
    fn refused_tracks_do_not_count_and_the_ranking_widens() {
        let v: Vec<f32> = (0..100).map(|i| i as f32).collect();
        // only multiples of 10 accepted: needs to look past the first chunk (k * 4)
        let top = mix_top_k_by(&[(1.0, v)], |_| false, 3, |i| i % 10 == 0);
        assert_eq!(
            top.iter().map(|x| x.0).collect::<Vec<_>>(),
            vec![90, 80, 70]
        );
    }

    #[test]
    fn bpm_is_half_double_time_aware() {
        assert_eq!(bpm_delta(70.0, 140.0), 0.0);
        assert_eq!(bpm_delta(120.0, 130.0), 10.0);
    }

    #[test]
    fn identical_track_scores_one_and_ranking_is_descending() {
        let ts = vec![
            T(vec![1.0, 0.0], vec![1.0, 0.0], 120.0),
            T(vec![1.0, 0.0], vec![1.0, 0.0], 120.0),
            T(vec![0.0, 1.0], vec![0.0, 1.0], 60.0),
            T(vec![0.6, 0.8], vec![1.0, 0.0], 125.0),
        ];
        let s = hybrid(&ts[0], &ts[1], &W);
        assert!((s.score - 1.0).abs() < 1e-6);
        let top: Vec<usize> = top_k(&ts, &ts[0], Some(0), &W, 2)
            .into_iter()
            .map(|(i, _)| i)
            .collect();
        assert_eq!(top, vec![1, 3]);
        let zero = Weights {
            w_audio: 0.0,
            w_text: 0.0,
            w_bpm: 0.0,
            bpm_tolerance: 20.0,
        };
        assert_eq!(hybrid(&ts[0], &ts[1], &zero).score, 0.0);
    }
}
