//! Demo catalogue (`seed/catalog.json`, shared with music-reco-hybrid-opti): metadata of the
//! synthetic clips plus the preset the `seed` binary renders them from.

use std::{fs::File, io::BufReader, path::Path};

use serde::Deserialize;

/// One entry of `seed/catalog.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct CatalogTrack {
    pub id: i64,
    pub title: String,
    pub artist: String,
    pub genre: String,
    pub subgenre: String,
    pub year: i64,
    pub bpm: f32,
    pub key: String,
    pub moods: Vec<String>,
    pub themes: Vec<String>,
    pub tags: Vec<String>,
    pub synth: Synth,
}

/// Rendering preset used by the `seed` binary.
#[derive(Debug, Clone, Deserialize)]
pub struct Synth {
    pub drums: String,
    pub lead: String,
    pub pad: String,
    pub bass: String,
    pub drive: f64,
    pub swing: f64,
    pub bright: f64,
    pub density: usize,
    pub noise: f64,
}

pub fn load_catalog(path: &Path) -> anyhow::Result<Vec<CatalogTrack>> {
    #[derive(Deserialize)]
    struct Catalog {
        tracks: Vec<CatalogTrack>,
    }
    Ok(serde_json::from_reader::<_, Catalog>(BufReader::new(File::open(path)?))?.tracks)
}
