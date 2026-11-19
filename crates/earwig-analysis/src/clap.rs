//! CLAP audio embedding in pure Rust: log-mel front-end (rustfft) + ONNX audio tower (ort).
//!
//! Model: `Xenova/clap-htsat-unfused` (ONNX export of `laion/clap-htsat-unfused`, the
//! same weights the Python repo uses through `transformers`). The front-end reproduces
//! HF `ClapFeatureExtractor` with `truncation="rand_trunc"` (centre crop here, for
//! determinism), `padding="repeatpad"`: 48 kHz, 10 s, n_fft 1024, hop 480, 64 Slaney mel
//! bands 50–14 000 Hz, power spectrum → dB. Output tensor [1, 1, 1001, 64].

use std::{path::Path, sync::Mutex};

use ort::{session::Session, value::Tensor};
use rustfft::{num_complex::Complex32, FftPlanner};

pub const SR: u32 = 48_000;
const LEN: usize = 480_000;
const N_FFT: usize = 1024;
const HOP: usize = 480;
const N_MELS: usize = 64;
const N_BINS: usize = N_FFT / 2 + 1;
const FRAMES: usize = LEN / HOP + 1;

fn hz_to_mel(f: f64) -> f64 {
    if f < 1000.0 {
        3.0 * f / 200.0
    } else {
        15.0 + (f / 1000.0).ln() * 27.0 / 6.4f64.ln()
    }
}
fn mel_to_hz(m: f64) -> f64 {
    if m < 15.0 {
        200.0 * m / 3.0
    } else {
        1000.0 * ((m - 15.0) * 6.4f64.ln() / 27.0).exp()
    }
}

/// Slaney-normalised triangular filters, `[N_MELS][N_BINS]` (= HF `mel_filter_bank`).
fn mel_filters() -> Vec<Vec<f32>> {
    let (lo, hi) = (hz_to_mel(50.0), hz_to_mel(14_000.0));
    let pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    (0..N_MELS)
        .map(|m| {
            let enorm = 2.0 / (pts[m + 2] - pts[m]);
            (0..N_BINS)
                .map(|k| {
                    let f = k as f64 * SR as f64 / 2.0 / (N_BINS - 1) as f64;
                    let down = (f - pts[m]) / (pts[m + 1] - pts[m]);
                    let up = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1]);
                    (down.min(up).max(0.0) * enorm) as f32
                })
                .collect()
        })
        .collect()
}

/// 48 kHz mono → log-mel `[FRAMES * N_MELS]`, row-major (frame, mel).
pub fn log_mel(x: &[f32]) -> Vec<f32> {
    // centre-crop / repeat-pad to exactly 10 s
    let mut w: Vec<f32> = if x.len() > LEN {
        let s = (x.len() - LEN) / 2;
        x[s..s + LEN].to_vec()
    } else {
        let mut v = x.repeat((LEN / x.len().max(1)).max(1));
        v.resize(LEN, 0.0);
        v
    };
    // reflect padding (center=True)
    let p = N_FFT / 2;
    let mut padded: Vec<f32> = (1..=p).rev().map(|i| w[i]).collect();
    padded.append(&mut w);
    let tail: Vec<f32> = (0..p).map(|j| padded[p + LEN - 2 - j]).collect();
    padded.extend(tail);

    let window: Vec<f32> = (0..N_FFT)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos())
        .collect();
    let filters = mel_filters();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N_FFT);
    let mut frame = vec![Complex32::default(); N_FFT];
    let mut power = vec![0f32; N_BINS];
    let mut out = Vec::with_capacity(FRAMES * N_MELS);
    for f in 0..FRAMES {
        for (i, c) in frame.iter_mut().enumerate() {
            *c = Complex32::new(padded[f * HOP + i] * window[i], 0.0);
        }
        fft.process(&mut frame);
        for (k, p) in power.iter_mut().enumerate() {
            *p = frame[k].norm_sqr();
        }
        out.extend(filters.iter().map(|filt| {
            let e: f32 = filt.iter().zip(&power).map(|(a, b)| a * b).sum();
            10.0 * e.max(1e-10).log10()
        }));
    }
    out
}

pub struct ClapAudio {
    // `Session::run` needs `&mut`; rayon workers share one session (ORT itself is multi-threaded).
    session: Mutex<Session>,
}

impl ClapAudio {
    pub fn load(model: &Path, env: &ort::environment::Environment) -> anyhow::Result<Self> {
        Ok(Self {
            session: Mutex::new(Session::builder(env)?.commit_from_file(model)?),
        })
    }

    /// Several ≤10 s windows of 48 kHz mono audio → one L2-normalised 512-d embedding
    /// (mean of the per-window embeddings, inferred as a single batch).
    pub fn embed(&self, windows: &[Vec<f32>]) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(!windows.is_empty(), "no audio");
        let mel: Vec<f32> = windows.iter().flat_map(|w| log_mel(w)).collect();
        let input = Tensor::from_array(([windows.len(), 1, FRAMES, N_MELS], mel))?;
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs!["input_features" => input])?;
        let (shape, data) = outputs["audio_embeds"].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            data.len() == 512 * windows.len(),
            "unexpected CLAP output shape {shape:?}"
        );
        let mut mean = vec![0f32; 512];
        for chunk in data.as_chunks::<512>().0 {
            let mut v = chunk.to_vec();
            earwig_core::l2_normalize(&mut v);
            mean.iter_mut().zip(&v).for_each(|(m, x)| *m += x);
        }
        earwig_core::l2_normalize(&mut mean);
        Ok(mean)
    }
}

/// CLAP text tower: puts words in the same 512-d space as the audio, which gives
/// zero-shot tagging ("is this closer to *jazz* or to *techno*?").
pub struct ClapText {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
}

impl ClapText {
    pub fn load(
        model: &Path,
        tokenizer: &Path,
        env: &ort::environment::Environment,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            session: Session::builder(env)?.commit_from_file(model)?,
            tokenizer: tokenizers::Tokenizer::from_file(tokenizer)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        })
    }

    /// L2-normalised 512-d embedding of a prompt (one at a time: the ONNX graph takes no attention mask).
    pub fn embed(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        let input = Tensor::from_array(([1usize, ids.len()], ids))?;
        let outputs = self.session.run(ort::inputs!["input_ids" => input])?;
        let (_, data) = outputs["text_embeds"].try_extract_tensor::<f32>()?;
        let mut v = data.to_vec();
        earwig_core::l2_normalize(&mut v);
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_shape_and_peak_band() {
        // 1 kHz sine: loudest mel band must be the one whose centre is closest to 1 kHz
        let x: Vec<f32> = (0..LEN)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SR as f32).sin())
            .collect();
        let mel = log_mel(&x);
        assert_eq!(mel.len(), FRAMES * N_MELS);
        let row = &mel[500 * N_MELS..501 * N_MELS];
        let argmax = (0..N_MELS)
            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
            .unwrap();
        let (lo, hi) = (hz_to_mel(50.0), hz_to_mel(14_000.0));
        let centre = mel_to_hz(lo + (hi - lo) * (argmax + 1) as f64 / (N_MELS + 1) as f64);
        assert!((850.0..1150.0).contains(&centre), "{centre}");
    }
}
