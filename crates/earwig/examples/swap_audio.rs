//! Benchmark another audio embedding with `earwig eval`: copy a library, swapping each track's
//! audio vector for an external one (e.g. Essentia's Discogs-EffNet). Tracks without one are
//! dropped, so both libraries can be evaluated on the same tracks.
//!
//!   cargo run --release -p earwig --example swap_audio -- FROM_STORE TO_STORE vectors.json
//!
//! With `CONCAT=1`, the external vector is appended to earwig's instead (both normalised).
//!
//! `vectors.json` maps media keys to vectors: `{"rap/1.mp3": [0.1, …], …}`.

use std::collections::HashMap;

use earwig::{library, storage::Store};
use earwig_core::l2_normalize;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [from, to, vectors] = args.as_slice() else {
        anyhow::bail!("usage: swap_audio FROM_STORE TO_STORE vectors.json");
    };
    let mut vectors: HashMap<String, Vec<f32>> = serde_json::from_slice(&std::fs::read(vectors)?)?;
    let mut index = library::require_latest(&Store::open(from)?).await?;
    let before = index.tracks.len();
    index
        .tracks
        .retain_mut(|t| match vectors.remove(&t.meta.path) {
            Some(mut v) => {
                l2_normalize(&mut v);
                // CONCAT=1: keep earwig's vector and append the external one, equal weight
                t.audio = if std::env::var_os("CONCAT").is_some() {
                    let mut both = [t.audio.as_slice(), &v].concat();
                    l2_normalize(&mut both);
                    both
                } else {
                    v
                };
                true
            }
            None => false,
        });
    println!(
        "{} of {before} tracks have an external vector",
        index.tracks.len()
    );
    library::publish(&Store::open(to)?, &library::new_run_id(), &index).await?;
    Ok(())
}
