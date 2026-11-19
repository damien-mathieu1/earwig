//! Everything that turns an audio file into searchable features.
//!
//! file ─► lofty tags
//!      ─► symphonia decode ─► stratum-dsp (BPM, key, grid stability) on the middle 60 s, RMS energy
//!                          ─► up to 3 × 10 s windows ─► rubato 48 kHz ─► log-mel ─► CLAP audio (ort) ─► mean 512-d
//!                                                                                   └► zero-shot genres / moods
//! metadata + auto-tags ─► text document ─► fastembed MiniLM ─► 384-d
//!
//! ```no_run
//! use earwig_analysis::{indexer, Analyzer};
//! use earwig_core::{scoring::{top_k, Weights}, Index};
//!
//! let dir = std::path::Path::new("/path/to/music");
//! let sources = indexer::local_sources(dir)?;
//! let local = |key: &str| Ok(indexer::LocalFile { path: dir.join(key), temporary: false });
//! let (tracks, _summary) = indexer::update(vec![], &sources, local, Analyzer::load, |_| {}, |_| {})?;
//! let index = Index { tracks, media: dir.display().to_string() };
//! index.save("music.index".as_ref())?;
//!
//! let (meta, query) = Analyzer::load()?.analyze_query("song.mp3".as_ref())?;
//! for (i, s) in top_k(&index.tracks, &query, None, &Weights::default(), 10) {
//!     println!("{:.3}  {}", s.score, index.tracks[i].meta.title);
//! }
//! # anyhow::Ok(())
//! ```

pub mod audio;
pub mod clap;
pub mod indexer;
pub mod tags;

use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};

use earwig_core::{l2_normalize, scoring::cosine, Query, TrackMeta};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use stratum_dsp::{analyze_audio, AnalysisConfig, Key};

/// Extensions picked up when scanning a folder or bucket (files without one are probed too).
pub const EXTENSIONS: &[&str] = &[
    "mp3", "flac", "wav", "ogg", "oga", "opus", "m4a", "mp4", "aac", "aif", "aiff", "caf", "mka",
    "webm",
];

/// Genre vocabulary for auto-tags and text-to-audio search.
pub const GENRES: &[&str] = &[
    "rock",
    "indie rock",
    "punk",
    "heavy metal",
    "pop",
    "synth-pop",
    "dream pop",
    "electronic",
    "house",
    "techno",
    "trance",
    "drum and bass",
    "dubstep",
    "ambient",
    "synthwave",
    "hip hop",
    "trap",
    "lo-fi hip hop",
    "R&B",
    "soul",
    "funk",
    "disco",
    "jazz",
    "swing",
    "bossa nova",
    "blues",
    "classical",
    "baroque",
    "solo piano",
    "orchestral",
    "soundtrack",
    "folk",
    "country",
    "reggae",
    "dub",
    "latin",
    "flamenco",
    "afrobeat",
    "gospel",
    "chanson",
];
/// Mood vocabulary for auto-tags and text-to-audio search.
pub const MOODS: &[&str] = &[
    "happy",
    "sad",
    "energetic",
    "calm",
    "relaxing",
    "dark",
    "aggressive",
    "romantic",
    "melancholic",
    "uplifting",
    "dreamy",
    "epic",
    "groovy",
    "peaceful",
    "tense",
    "nostalgic",
    "playful",
    "mysterious",
    "sensual",
    "powerful",
    "hypnotic",
    "joyful",
    "angry",
    "chill",
];
const N_AUTO_GENRES: usize = 2;
const N_AUTO_MOODS: usize = 3;
const NOTES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// `$EARWIG_CACHE`, else `~/.cache/earwig`: models are downloaded there once.
pub fn cache_dir() -> PathBuf {
    std::env::var_os("EARWIG_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| ".".into()))
                .join(".cache/earwig")
        })
}

/// (size, mtime) — cheap change detection for incremental indexing.
pub fn fingerprint(path: &Path) -> std::io::Result<(u64, u64)> {
    let m = std::fs::metadata(path)?;
    let mtime = m
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    Ok((m.len(), mtime))
}

/// One ONNX Runtime environment per process (ORT refuses a second one), shared by every model.
fn ort_env() -> anyhow::Result<ort::environment::Environment> {
    static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
    if let Some(e) = ENV.get() {
        return Ok(e.clone());
    }
    let e = ort::init().build()?;
    Ok(ENV.get_or_init(|| e).clone())
}

fn clap_repo() -> anyhow::Result<hf_hub::api::sync::ApiRepo> {
    let hf = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir().join("hf"))
        .build()?;
    Ok(hf.model("Xenova/clap-htsat-unfused".into()))
}

/// CLAP text tower + the embedded genre / mood vocabularies: words in the same space as the audio.
/// Light (~120 MB): enough to search an index by mood or style without the audio models.
pub struct ClapVocab {
    clap_text: Mutex<clap::ClapText>,
    /// one vector per entry of [`GENRES`] / [`MOODS`]
    pub genres: Vec<Vec<f32>>,
    pub moods: Vec<Vec<f32>>,
}

impl ClapVocab {
    pub fn load() -> anyhow::Result<Self> {
        let repo = clap_repo()?;
        let clap_text = clap::ClapText::load(
            &repo.get("onnx/text_model_quantized.onnx")?,
            &repo.get("tokenizer.json")?,
            &ort_env()?,
        )?;
        let mut v = Self {
            clap_text: Mutex::new(clap_text),
            genres: vec![],
            moods: vec![],
        };
        v.genres = GENRES
            .iter()
            .map(|l| v.embed_label(l, "genre"))
            .collect::<anyhow::Result<_>>()?;
        v.moods = MOODS
            .iter()
            .map(|l| v.embed_label(l, "mood"))
            .collect::<anyhow::Result<_>>()?;
        Ok(v)
    }

    /// Prompt ensembling: a single template makes a few labels ("bossa nova") win on almost any track.
    /// `kind` is "genre" or "mood"; `label` can be any English words ("upbeat summer").
    pub fn embed_label(&self, label: &str, kind: &str) -> anyhow::Result<Vec<f32>> {
        let mut clap_text = self.clap_text.lock().unwrap();
        let mut sum = vec![0f32; 512];
        for p in [
            format!("This is {label} music."),
            format!("A {label} song."),
            format!("{label} music"),
            format!("The {kind} of this song is {label}."),
        ] {
            sum.iter_mut()
                .zip(clap_text.embed(&p)?)
                .for_each(|(s, x)| *s += x);
        }
        l2_normalize(&mut sum);
        Ok(sum)
    }
}

pub struct Analyzer {
    clap: clap::ClapAudio,
    pub vocab: ClapVocab,
    text: Mutex<TextEmbedding>,
}

impl Analyzer {
    /// Loads (and on first run downloads, ~700 MB) the CLAP audio + text towers and MiniLM.
    pub fn load() -> anyhow::Result<Self> {
        let env = ort_env()?;
        let clap = clap::ClapAudio::load(&clap_repo()?.get("onnx/audio_model.onnx")?, &env)?;
        let vocab = ClapVocab::load()?;
        let text = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::ParaphraseMLMiniLML12V2)
                .with_cache_dir(cache_dir().join("fastembed"))
                .with_environment(env),
        )?;
        Ok(Self {
            clap,
            vocab,
            text: Mutex::new(text),
        })
    }

    /// Audio + tags → metadata (id 0, no text yet) and the CLAP embedding.
    pub fn analyze(&self, path: &Path) -> anyhow::Result<(TrackMeta, Vec<f32>)> {
        let t = tags::read(path);
        let (pcm, sr) = audio::decode_mono(path)?;
        anyhow::ensure!(pcm.len() > sr as usize, "shorter than 1 s");

        // MIR on the middle minute: enough for a stable tempo, bounded cost on long tracks
        let mir_len = (60 * sr as usize).min(pcm.len());
        let mid = (pcm.len() - mir_len) / 2;
        let mir = analyze_audio(&pcm[mid..mid + mir_len], sr, AnalysisConfig::default())
            .map_err(|e| anyhow::anyhow!("stratum-dsp: {e:?}"))?;
        let energy =
            (pcm.iter().map(|x| (x * x) as f64).sum::<f64>() / pcm.len() as f64).sqrt() as f32;

        let windows = windows(&pcm, sr)
            .into_iter()
            .map(|w| audio::resample(w, sr, clap::SR))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let emb = self.clap.embed(&windows)?;

        let meta = TrackMeta {
            path: path.display().to_string(),
            title: t.title.unwrap_or_else(|| {
                path.file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            }),
            artist: t.artist.unwrap_or_default(),
            album: t.album.unwrap_or_default(),
            genre: t.genre.unwrap_or_default(),
            year: t.year,
            bpm: mir.bpm,
            key: match mir.key {
                Key::Major(i) => format!("{} major", NOTES[i as usize % 12]),
                Key::Minor(i) => format!("{} minor", NOTES[i as usize % 12]),
            },
            danceability: mir.grid_stability,
            energy,
            duration_s: pcm.len() as f32 / sr as f32,
            auto_genres: top_labels(&emb, &self.vocab.genres, GENRES, N_AUTO_GENRES),
            auto_moods: top_labels(&emb, &self.vocab.moods, MOODS, N_AUTO_MOODS),
            ..Default::default()
        };
        Ok((meta, emb))
    }

    /// One external file → its metadata and the features to query an index with.
    pub fn analyze_query(&self, path: &Path) -> anyhow::Result<(TrackMeta, Query)> {
        let (meta, audio) = self.analyze(path)?;
        let text = self.embed_documents(&[meta.text_document()])?.remove(0);
        let bpm = meta.bpm;
        Ok((meta, Query { audio, text, bpm }))
    }

    /// Batched MiniLM embeddings (L2-normalised) of `TrackMeta::text_document()`s.
    pub fn embed_documents(&self, docs: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut out = self.text.lock().unwrap().embed(docs, Some(64))?;
        out.iter_mut().for_each(|v| l2_normalize(v));
        Ok(out)
    }
}

/// Up to 3 windows of ≤10 s centred at 1/6, 1/2, 5/6 of the track (or 1/4, 3/4; or the whole clip).
fn windows(pcm: &[f32], sr: u32) -> Vec<&[f32]> {
    let win = 10 * sr as usize;
    if pcm.len() <= win {
        return vec![pcm];
    }
    let n = pcm.len().div_ceil(win).min(3);
    (0..n)
        .map(|i| {
            let start = (pcm.len() * (2 * i + 1) / (2 * n))
                .saturating_sub(win / 2)
                .min(pcm.len() - win);
            &pcm[start..start + win]
        })
        .collect()
}

fn top_labels(emb: &[f32], vecs: &[Vec<f32>], labels: &[&str], n: usize) -> Vec<String> {
    let mut idx: Vec<usize> = (0..labels.len()).collect();
    idx.sort_by(|&a, &b| cosine(emb, &vecs[b]).total_cmp(&cosine(emb, &vecs[a])));
    idx.into_iter()
        .take(n)
        .map(|i| labels[i].to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn windows_cover_the_track() {
        let sr = 100;
        assert_eq!(super::windows(&vec![0.0; 500], sr).len(), 1);
        let x: Vec<f32> = (0..6000).map(|i| i as f32).collect(); // 60 s
        let w = super::windows(&x, sr);
        assert_eq!(w.len(), 3);
        assert!(w.iter().all(|w| w.len() == 1000));
        assert_eq!((w[0][0], w[1][0], w[2][0]), (500.0, 2500.0, 4500.0));
    }
}
