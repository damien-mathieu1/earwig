//! Tempo accuracy of stratum-dsp settings against a catalogue with ground-truth BPM.
//!
//!   cargo run --release -p earwig --example bpm_bench -- seed/catalog.json seed/audio
//!
//! Accuracy1: within ±4 % of the label. Accuracy2: same, also accepting ×2, ×½, ×3, ×⅓ (MIREX).

use std::path::Path;

use earwig_analysis::audio::decode_mono;
use rayon::prelude::*;
use stratum_dsp::{analyze_audio, AnalysisConfig};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (catalog, dir) = (Path::new(&args[1]), Path::new(&args[2]));
    let tracks = earwig::catalog::load_catalog(catalog)?;
    let audio: Vec<(f32, Vec<f32>, u32)> = tracks
        .par_iter()
        .map(|t| decode_mono(&dir.join(format!("{:04}.wav", t.id))).map(|(x, sr)| (t.bpm, x, sr)))
        .collect::<anyhow::Result<_>>()?;

    let d = AnalysisConfig::default;
    let configs: Vec<(&str, AnalysisConfig)> = vec![
        ("default", d()),
        (
            "bpm_fusion",
            AnalysisConfig {
                enable_bpm_fusion: true,
                ..d()
            },
        ),
        (
            "human_prior",
            AnalysisConfig {
                tempogram_multi_res_use_human_prior: true,
                ..d()
            },
        ),
        (
            "fusion+prior",
            AnalysisConfig {
                enable_bpm_fusion: true,
                tempogram_multi_res_use_human_prior: true,
                ..d()
            },
        ),
        (
            "legacy",
            AnalysisConfig {
                force_legacy_bpm: true,
                ..d()
            },
        ),
        (
            "hpss_onsets",
            AnalysisConfig {
                enable_hpss_onsets: true,
                ..d()
            },
        ),
    ];
    let close = |a: f32, b: f32| (a - b).abs() <= 0.04 * b;
    println!("{:<14} {:>6} {:>6}", "config", "acc1", "acc2");
    for (name, cfg) in configs {
        let (a1, a2) = audio
            .par_iter()
            .map(|(label, x, sr)| {
                let bpm = analyze_audio(x, *sr, cfg.clone())
                    .map(|r| r.bpm)
                    .unwrap_or(0.0);
                let ok1 = close(bpm, *label);
                let ok2 = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]
                    .iter()
                    .any(|f| close(bpm, label * f));
                (ok1 as u32, ok2 as u32)
            })
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        println!(
            "{name:<14} {:>5.1}% {:>5.1}%",
            100.0 * a1 as f32 / audio.len() as f32,
            100.0 * a2 as f32 / audio.len() as f32
        );
    }
    Ok(())
}
