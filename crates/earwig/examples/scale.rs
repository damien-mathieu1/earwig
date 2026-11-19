//! Query latency of the exact scan vs catalogue size, on random unit vectors.
//!
//!   cargo run --release -p earwig --example scale

use std::time::Instant;

use earwig_core::{
    l2_normalize,
    scoring::{top_k, Weights},
    Track, TrackMeta,
};

fn random_tracks(n: usize) -> Vec<Track> {
    let mut x: u32 = 0x9E37_79B9;
    let mut rnd = move || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x as f32 / u32::MAX as f32 - 0.5
    };
    (0..n)
        .map(|i| {
            let mut audio: Vec<f32> = (0..512).map(|_| rnd()).collect();
            let mut text: Vec<f32> = (0..384).map(|_| rnd()).collect();
            l2_normalize(&mut audio);
            l2_normalize(&mut text);
            Track {
                meta: TrackMeta {
                    id: i as i64,
                    bpm: 60.0 + 120.0 * (rnd() + 0.5),
                    ..Default::default()
                },
                audio,
                text,
                fingerprint: (0, 0),
            }
        })
        .collect()
}

fn main() {
    let w = Weights::default();
    for n in [10_000, 100_000, 1_000_000] {
        let tracks = random_tracks(n);
        let mb = n as f64 * (512.0 + 384.0) * 4.0 / 1e6;
        let reps = (2_000_000 / n).clamp(3, 200);
        let t0 = Instant::now();
        for q in 0..reps {
            std::hint::black_box(top_k(&tracks, &tracks[q], Some(q), &w, 10));
        }
        println!(
            "{n:>9} tracks  {mb:>7.0} MB vectors  {:>8.2} ms/query",
            t0.elapsed().as_secs_f64() * 1e3 / reps as f64
        );
    }
}
