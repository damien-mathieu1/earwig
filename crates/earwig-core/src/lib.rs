//! Core of the `earwig` engine: track metadata, persisted index, hybrid scoring.
//! No ML dependency — enough to load an index and serve recommendations.
//!
//! ```no_run
//! use earwig_core::{scoring::{top_k, Weights}, Index};
//! let index = Index::load("data/index.bin".as_ref())?;
//! let src = 0; // position of the query track
//! for (i, s) in top_k(&index.tracks, &index.tracks[src], Some(src), &Weights::default(), 10) {
//!     println!("{:.3}  {}", s.score, index.tracks[i].meta.title);
//! }
//! # anyhow::Ok(())
//! ```

pub mod scoring;

use std::path::Path;

use anyhow::Context;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrackMeta {
    pub id: i64,
    pub path: String,
    pub title: String,
    pub artist: String,
    pub album: String,
    pub genre: String,
    pub subgenre: String,
    pub year: Option<i64>,
    /// ground truth from a catalogue, when there is one
    pub bpm_label: Option<f32>,
    pub key_label: Option<String>,
    /// detected by stratum-dsp
    pub bpm: f32,
    pub key: String,
    /// stratum-dsp beat-grid stability, used as a danceability proxy
    pub danceability: f32,
    /// RMS
    pub energy: f32,
    pub duration_s: f32,
    /// human metadata (catalogue)
    pub moods: Vec<String>,
    pub themes: Vec<String>,
    pub tags: Vec<String>,
    /// CLAP zero-shot tags, always computed from the audio
    pub auto_genres: Vec<String>,
    pub auto_moods: Vec<String>,
    /// effective analysis cost of this track (batch wall time / batch size)
    pub ingest_ms: f32,
}

impl TrackMeta {
    /// The text that gets embedded. Human metadata wins; CLAP auto-tags fill the gaps,
    /// so a folder of untagged files still gets a meaningful text vector.
    /// (For catalogue tracks this is byte-identical to music-reco-hybrid-opti's document.)
    pub fn text_document(&self) -> String {
        let human: Vec<&str> = [&self.genre, &self.subgenre]
            .into_iter()
            .map(String::as_str)
            .filter(|s| !s.is_empty())
            .collect();
        let genres = if human.is_empty() {
            self.auto_genres.join(", ")
        } else {
            human.join(", ")
        };
        let moods = if self.moods.is_empty() {
            &self.auto_moods
        } else {
            &self.moods
        };
        let mut doc = format!("{genres}.");
        for (label, items) in [
            ("Ambiances", moods),
            ("Thèmes", &self.themes),
            ("Tags", &self.tags),
        ] {
            if !items.is_empty() {
                doc += &format!(" {label} : {}.", items.join(", "));
            }
        }
        doc
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    pub meta: TrackMeta,
    /// CLAP audio embedding (512-d, L2-normalised, mean of up to 3 windows)
    pub audio: Vec<f32>,
    /// MiniLM text embedding (384-d, L2-normalised)
    pub text: Vec<f32>,
    /// (size, mtime) of the file when it was analysed — drives incremental re-indexing
    pub fingerprint: (u64, u64),
}

impl scoring::Scorable for Track {
    fn audio(&self) -> &[f32] {
        &self.audio
    }
    fn text(&self) -> &[f32] {
        &self.text
    }
    fn bpm(&self) -> f32 {
        self.meta.bpm
    }
}

/// A library snapshot: what `earwig index` publishes and `earwig serve` / `similar` / `eval` load.
/// Track paths are keys relative to `media` (a local directory or an `s3://…` URL), so a snapshot
/// stays valid on every machine that can reach the media.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Index {
    pub tracks: Vec<Track>,
    pub media: String,
}

/// Snapshot header: format 1 added `album`. Bytes without it are the original format 0.
const MAGIC: &[u8] = b"EARWIG\x001";

impl Index {
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut out = MAGIC.to_vec();
        bincode::serialize_into(&mut out, self)?;
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let index = match bytes.strip_prefix(MAGIC) {
            Some(rest) => bincode::deserialize(rest),
            None => bincode::deserialize::<legacy::IndexV0>(bytes).map(Into::into),
        };
        index.map_err(|e| anyhow::anyhow!("unreadable index or index from a newer version ({e})"))
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        // write-then-rename so an interrupted run never leaves a truncated index
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, self.to_bytes()?)?;
        Ok(std::fs::rename(tmp, path)?)
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
            .with_context(|| format!("loading {}", path.display()))
    }
}

/// Format 0 (no header, no album), read so old snapshots keep working.
mod legacy {
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct TrackMetaV0 {
        id: i64,
        path: String,
        title: String,
        artist: String,
        genre: String,
        subgenre: String,
        year: Option<i64>,
        bpm_label: Option<f32>,
        key_label: Option<String>,
        bpm: f32,
        key: String,
        danceability: f32,
        energy: f32,
        duration_s: f32,
        moods: Vec<String>,
        themes: Vec<String>,
        tags: Vec<String>,
        auto_genres: Vec<String>,
        auto_moods: Vec<String>,
        ingest_ms: f32,
    }

    #[derive(Deserialize)]
    struct TrackV0 {
        meta: TrackMetaV0,
        audio: Vec<f32>,
        text: Vec<f32>,
        fingerprint: (u64, u64),
    }

    #[derive(Deserialize)]
    pub(super) struct IndexV0 {
        tracks: Vec<TrackV0>,
        media: String,
    }

    impl From<IndexV0> for super::Index {
        fn from(v: IndexV0) -> Self {
            let tracks = v
                .tracks
                .into_iter()
                .map(|t| {
                    let m = t.meta;
                    super::Track {
                        meta: super::TrackMeta {
                            id: m.id,
                            path: m.path,
                            title: m.title,
                            artist: m.artist,
                            album: String::new(),
                            genre: m.genre,
                            subgenre: m.subgenre,
                            year: m.year,
                            bpm_label: m.bpm_label,
                            key_label: m.key_label,
                            bpm: m.bpm,
                            key: m.key,
                            danceability: m.danceability,
                            energy: m.energy,
                            duration_s: m.duration_s,
                            moods: m.moods,
                            themes: m.themes,
                            tags: m.tags,
                            auto_genres: m.auto_genres,
                            auto_moods: m.auto_moods,
                            ingest_ms: m.ingest_ms,
                        },
                        audio: t.audio,
                        text: t.text,
                        fingerprint: t.fingerprint,
                    }
                })
                .collect();
            Self {
                tracks,
                media: v.media,
            }
        }
    }
}

/// Stable id of a media key: the same file gets the same id in every run and every shard,
/// so shards never collide and ids survive re-indexing. FNV-1a, kept below 2^53 for JavaScript.
pub fn track_id(key: &str) -> i64 {
    let h = key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    });
    (h & ((1 << 53) - 1)) as i64
}

/// Features of a track that is not in the index (a file passed to `earwig similar`, an upload…).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub audio: Vec<f32>,
    pub text: Vec<f32>,
    pub bpm: f32,
}

impl scoring::Scorable for Query {
    fn audio(&self) -> &[f32] {
        &self.audio
    }
    fn text(&self) -> &[f32] {
        &self.text
    }
    fn bpm(&self) -> f32 {
        self.bpm
    }
}

pub fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_ids_are_stable_and_js_safe() {
        assert_eq!(track_id("rap/1171140.mp3"), track_id("rap/1171140.mp3"));
        assert_ne!(track_id("a.mp3"), track_id("b.mp3"));
        assert!((0..1000).all(|i| (0..1 << 53).contains(&track_id(&format!("{i}.mp3")))));
        let idx = Index {
            tracks: vec![],
            media: "s3://b/music".into(),
        };
        assert_eq!(
            Index::from_bytes(&idx.to_bytes().unwrap()).unwrap().media,
            "s3://b/music"
        );
    }

    #[test]
    fn document_prefers_human_metadata_and_falls_back_to_auto_tags() {
        let mut m = TrackMeta {
            genre: "rock".into(),
            subgenre: "punk".into(),
            moods: vec!["rageur".into()],
            themes: vec!["révolte".into()],
            tags: vec!["DIY".into()],
            auto_genres: vec!["metal".into()],
            auto_moods: vec!["angry".into()],
            ..Default::default()
        };
        assert_eq!(
            m.text_document(),
            "rock, punk. Ambiances : rageur. Thèmes : révolte. Tags : DIY."
        );
        m = TrackMeta {
            auto_genres: m.auto_genres,
            auto_moods: m.auto_moods,
            ..Default::default()
        };
        assert_eq!(m.text_document(), "metal. Ambiances : angry.");
    }
}
