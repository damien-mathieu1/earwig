//! Multi-style synthesiser for the seed dataset — 1:1 port of
//! music-reco-hybrid-opti/ingestion/seed.py, so both repos use the same clips.
//!
//! Each clip (15 s, 22.05 kHz mono) is rendered from the track's BPM, key and `synth`
//! preset: chord progression (I-V-vi-IV / i-VI-III-VII), pad, bass, arpeggiated lead,
//! drum pattern, swing, vinyl noise, one-pole low-pass ("brightness") and tanh drive.

use std::f64::consts::PI;

use crate::catalog::CatalogTrack;

pub const SR: u32 = 22050;
pub const DUR: f64 = 15.0;
const N: usize = (SR as f64 * DUR) as usize;
const NOTES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// 16 steps per bar: (kick, snare, hat)
fn pattern(name: &str) -> (&'static str, &'static str, &'static str) {
    match name {
        "four_floor" => ("x...x...x...x...", "....x.......x...", "..x...x...x...x."),
        "techno" => ("x...x...x...x...", "....x.......x...", "xxxxxxxxxxxxxxxx"),
        "disco" => ("x...x...x...x...", "....x.......x...", "x.x.x.x.x.x.x.x."),
        "breakbeat" => ("x.....x...x.....", "....x.......x..x", "x.x.x.x.x.x.x.x."),
        "dnb" => ("x.........x.....", "....x.......x...", "x.x.x.x.x.x.x.x."),
        "boombap" => ("x......x..x.....", "....x.......x...", "x.x.x.x.x.x.x.x."),
        "trap" => ("x......x........", "........x.......", "xxxxxxxxxxxxxxxx"),
        "halftime" => ("x.........x.....", "........x.......", "x.x.x.x.x.x.x.x."),
        "rock" => ("x.......x.x.....", "....x.......x...", "x.x.x.x.x.x.x.x."),
        "punk" => ("x...x...x...x...", "..x...x...x...x.", "x.x.x.x.x.x.x.x."),
        "swing" => ("x.......x.......", "....x.......x...", "x...x.x.x...x.x."),
        "bossa" => ("x..xx..xx..xx..x", "x..x..x...x..x..", "x.x.x.x.x.x.x.x."),
        "onedrop" => ("........x.......", "........x.......", "..x...x...x...x."),
        "afro" => ("x.....x.x.....x.", "....x..x....x...", "xxxxxxxxxxxxxxxx"),
        _ => ("", "", ""),
    }
}

/// xorshift32 white noise, identical to the Python generator.
fn noise_table(track_id: i64) -> Vec<f64> {
    let mut x = (track_id as u32).wrapping_mul(2654435761) ^ 0x9E37_79B9;
    if x == 0 {
        x = 1;
    }
    (0..16384)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as f64 / 4294967296.0 * 2.0 - 1.0
        })
        .collect()
}

fn midi_hz(m: f64) -> f64 {
    440.0 * 2f64.powf((m - 69.0) / 12.0)
}

fn osc(kind: &str, f: f64, t: f64) -> f64 {
    let p = f * t;
    let s = |h: f64| (2.0 * PI * p * h).sin();
    match kind {
        "sine" => s(1.0),
        "saw" | "pluck" => 2.0 * p.rem_euclid(1.0) - 1.0,
        "square" => {
            if p.rem_euclid(1.0) < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        "piano" => (s(1.0) + 0.5 * s(2.0) + 0.25 * s(3.0)) / 1.75,
        "organ" => (s(1.0) + 0.5 * s(2.0) + 0.3 * s(3.0)) / 1.8,
        other => panic!("unknown oscillator {other}"),
    }
}

fn note(buf: &mut [f64], kind: &str, midi: f64, start: f64, dur: f64, amp: f64) {
    let i0 = (start * SR as f64) as usize;
    let n = ((dur * SR as f64) as usize).min(N.saturating_sub(i0));
    let f = midi_hz(midi);
    let decays = matches!(kind, "pluck" | "piano");
    for (j, out) in buf[i0.min(N)..][..n].iter_mut().enumerate() {
        let t = j as f64 / SR as f64;
        let mut env = (t / 0.005).min(1.0) * ((dur - t) / 0.01).clamp(0.0, 1.0);
        if decays {
            env *= (-4.0 * t).exp();
        }
        *out += amp * env * osc(kind, f, t);
    }
}

fn drum(buf: &mut [f64], kind: char, start: f64, noise: &[f64]) {
    let i0 = (start * SR as f64) as usize;
    let len = match kind {
        'k' => 0.25,
        's' => 0.2,
        _ => 0.05,
    };
    let n = ((len * SR as f64) as usize).min(N.saturating_sub(i0));
    for (j, out) in buf[i0.min(N)..][..n].iter_mut().enumerate() {
        let t = j as f64 / SR as f64;
        *out += match kind {
            'k' => {
                0.9 * (2.0 * PI * (50.0 * t + 100.0 * (1.0 - (-30.0 * t).exp()) / 30.0)).sin()
                    * (-12.0 * t).exp()
            }
            's' => 0.5 * (0.6 * noise[j] + 0.4 * (2.0 * PI * 180.0 * t).sin()) * (-20.0 * t).exp(),
            _ => 0.25 * (noise[j + 1] - noise[j]) * (-60.0 * t).exp(),
        };
    }
}

pub fn render(track: &CatalogTrack) -> Vec<f32> {
    let s = &track.synth;
    let (tonic_name, mode) = track.key.split_once(' ').expect("key like 'A minor'");
    let tonic = 48
        + NOTES
            .iter()
            .position(|&n| n == tonic_name)
            .expect("note name") as i32;
    let (scale, prog): ([i32; 7], [i32; 4]) = if mode == "major" {
        ([0, 2, 4, 5, 7, 9, 11], [0, 4, 5, 3])
    } else {
        ([0, 2, 3, 5, 7, 8, 10], [0, 5, 2, 6])
    };
    let deg = |d: i32| scale[d.rem_euclid(7) as usize] + 12 * d.div_euclid(7);
    let beat = 60.0 / track.bpm as f64;
    let step = beat / 4.0;
    let n_steps = (DUR / step) as usize;
    let noise = noise_table(track.id);
    let mut buf = vec![0f64; N];
    let (kick, snare, hat) = pattern(&s.drums);

    for st in 0..n_steps {
        let (bar, pos) = (st / 16, st % 16);
        let chord: Vec<f64> = [0, 2, 4]
            .iter()
            .map(|&k| (tonic + deg(prog[bar % 4] + k)) as f64)
            .collect();
        let t = st as f64 * step
            + if pos % 4 == 2 {
                s.swing * beat * 0.5
            } else {
                0.0
            };
        if pos == 0 && s.pad != "none" {
            for &m in &chord {
                note(&mut buf, &s.pad, m, t, 16.0 * step, 0.08);
            }
        }
        if pos % 4 == 0 && s.bass != "none" {
            let m = chord[0] - 12.0
                + if s.bass == "pluck" && (pos / 4) % 2 == 1 {
                    7.0
                } else {
                    0.0
                };
            if s.bass == "saw" {
                note(&mut buf, "saw", m, t, 2.0 * step, 0.3);
                note(&mut buf, "saw", m, t + 2.0 * step, 2.0 * step, 0.3);
            } else {
                note(
                    &mut buf,
                    if s.bass == "sub" { "sine" } else { "pluck" },
                    m,
                    t,
                    4.0 * step,
                    0.35,
                );
            }
        }
        if s.density > 0 && pos % (4 / s.density) == 0 {
            let k = pos / (4 / s.density);
            note(
                &mut buf,
                &s.lead,
                chord[k % 3] + 12.0 * (1 + (k / 3) % 2) as f64,
                t,
                (4 / s.density) as f64 * step,
                0.25,
            );
        }
        for (pat, kind) in [(kick, 'k'), (snare, 's'), (hat, 'h')] {
            if pat.as_bytes().get(pos) == Some(&b'x') {
                drum(&mut buf, kind, t, &noise);
            }
        }
    }

    if s.noise > 0.0 {
        for (i, x) in buf.iter_mut().enumerate() {
            *x += s.noise * noise[(i * 7) % noise.len()];
        }
    }
    let a = 0.05 + 0.9 * s.bright; // one-pole low-pass = "brightness"
    let mut y = 0.0;
    for x in buf.iter_mut() {
        y = a * *x + (1.0 - a) * y;
        *x = y;
    }
    if s.drive > 0.0 {
        let g = 1.0 + s.drive * 8.0;
        buf.iter_mut().for_each(|x| *x = (*x * g).tanh() / g.tanh());
    }
    let peak = buf.iter().fold(1e-9f64, |m, x| m.max(x.abs()));
    buf.iter().map(|x| (x * 0.9 / peak) as f32).collect()
}
