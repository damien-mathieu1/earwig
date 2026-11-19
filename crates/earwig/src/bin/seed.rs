//! `cargo run --release --bin seed` — render one WAV per track of seed/catalog.json
//! into seed/audio/ (offline, no network).

use std::path::Path;

use earwig::synth;
use rayon::prelude::*;

fn main() -> anyhow::Result<()> {
    let catalog = earwig::catalog::load_catalog(Path::new("seed/catalog.json"))?;
    let out = Path::new("seed/audio");
    std::fs::create_dir_all(out)?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: synth::SR,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    catalog.par_iter().try_for_each(|t| -> anyhow::Result<()> {
        let mut w = hound::WavWriter::create(out.join(format!("{:04}.wav", t.id)), spec)?;
        for s in synth::render(t) {
            w.write_sample((s * 32767.0) as i16)?;
        }
        Ok(w.finalize()?)
    })?;
    println!("{} clips -> {}", catalog.len(), out.display());
    Ok(())
}
