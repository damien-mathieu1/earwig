//! Incremental, shardable indexing of a media collection (a local folder, a bucket…).
//!
//! The caller lists the media as [`Source`]s (key + fingerprint) and says how to get a local
//! file for a key; unchanged tracks of the previous snapshot are kept as they are, new or
//! modified ones are analysed in parallel, the rest (deleted files) is dropped. Track ids are
//! derived from the key ([`earwig_core::track_id`]), so independent shards never collide.

use std::path::{Path, PathBuf};

use earwig_core::{track_id, Track, TrackMeta};
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

use crate::{Analyzer, EXTENSIONS};

/// One media file: its key relative to the media root and a change detector (size, mtime).
#[derive(Debug, Clone)]
pub struct Source {
    pub key: String,
    pub fingerprint: (u64, u64),
}

/// A local copy of a media file; temporary copies are deleted when dropped.
pub struct LocalFile {
    pub path: PathBuf,
    pub temporary: bool,
}

impl Drop for LocalFile {
    fn drop(&mut self) {
        if self.temporary {
            let _ = std::fs::remove_file(&self.path);
            if let Some(dir) = self.path.parent() {
                let _ = std::fs::remove_dir(dir);
            }
        }
    }
}

/// Progress notifications (called from worker threads).
#[derive(Debug)]
pub enum Event<'a> {
    /// Models are loaded and `files` files are about to be analysed.
    Started {
        files: usize,
    },
    Analysed {
        key: &'a str,
        meta: &'a TrackMeta,
    },
    Skipped {
        key: &'a str,
        error: &'a anyhow::Error,
    },
}

#[derive(Debug, Default)]
pub struct Summary {
    pub tracks: usize,
    pub analysed: usize,
    pub failed: usize,
    pub removed: usize,
    /// effective cost per analysed track (batch wall time / batch size)
    pub ms_per_track: f32,
}

/// A known audio extension, or none at all (apps often store files under an id): the
/// decoder then recognises the format from the content. Hidden files are skipped.
pub fn is_audio(key: &str) -> bool {
    let path = Path::new(key);
    if path
        .file_name()
        .is_none_or(|n| n.to_string_lossy().starts_with('.'))
    {
        return false;
    }
    match path.extension().and_then(|e| e.to_str()) {
        None => true,
        Some(e) => EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()),
    }
}

/// Shard of a key among `n` (stable: same key, same shard, on every machine).
pub fn shard_of(key: &str, n: usize) -> usize {
    (track_id(key) as u64 % n.max(1) as u64) as usize
}

/// Audio files under a local directory, keys relative to it (sorted).
pub fn local_sources(dir: &Path) -> anyhow::Result<Vec<Source>> {
    let root = dir.canonicalize()?;
    let mut out = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d)? {
            let p = entry?.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let key = p.strip_prefix(&root)?.to_string_lossy().replace('\\', "/");
            if is_audio(&key) {
                out.push(Source {
                    key,
                    fingerprint: crate::fingerprint(&p)?,
                });
            }
        }
    }
    out.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(out)
}

/// Brings a set of tracks up to date with `sources`. Returns the new set, sorted by id.
///
/// * `fetch` gives a local file for a key (the file itself, or a temporary download).
/// * `analyzer` is only built (by calling it) when at least one file needs analysis.
/// * `edit_meta` can overwrite metadata (title, genre, moods…) before the text embedding.
/// * `on_event` receives progress notifications.
pub fn update(
    previous: Vec<Track>,
    sources: &[Source],
    fetch: impl Fn(&str) -> anyhow::Result<LocalFile> + Sync,
    analyzer: impl FnOnce() -> anyhow::Result<Analyzer>,
    edit_meta: impl Fn(&mut TrackMeta),
    on_event: impl Fn(Event) + Sync,
) -> anyhow::Result<(Vec<Track>, Summary)> {
    let mut old: HashMap<String, Track> = previous
        .into_iter()
        .map(|t| (t.meta.path.clone(), t))
        .collect();
    let (mut keep, mut todo) = (Vec::new(), Vec::new());
    for s in sources {
        match old.remove(&s.key) {
            Some(t) if t.fingerprint == s.fingerprint => keep.push(t),
            _ => todo.push(s),
        }
    }
    let mut summary = Summary {
        removed: old.len(),
        ..Default::default()
    };

    if !todo.is_empty() {
        let analyzer = analyzer()?;
        on_event(Event::Started { files: todo.len() });
        let t0 = Instant::now();
        let analysed: Vec<_> = todo
            .par_iter()
            .filter_map(
                |s| match fetch(&s.key).and_then(|f| analyzer.analyze(&f.path)) {
                    Ok((mut meta, audio)) => {
                        meta.path = s.key.clone();
                        meta.id = track_id(&s.key);
                        on_event(Event::Analysed {
                            key: &s.key,
                            meta: &meta,
                        });
                        Some((meta, audio, s.fingerprint))
                    }
                    Err(error) => {
                        on_event(Event::Skipped {
                            key: &s.key,
                            error: &error,
                        });
                        None
                    }
                },
            )
            .collect();
        summary.failed = todo.len() - analysed.len();

        let (mut metas, mut rest) = (Vec::new(), Vec::new());
        for (mut m, audio, fp) in analysed {
            edit_meta(&mut m);
            metas.push(m);
            rest.push((audio, fp));
        }
        let docs: Vec<String> = metas.iter().map(|m| m.text_document()).collect();
        let texts = analyzer.embed_documents(&docs)?;
        summary.analysed = metas.len();
        summary.ms_per_track =
            (t0.elapsed().as_secs_f64() * 1e3 / metas.len().max(1) as f64) as f32;
        keep.extend(metas.into_iter().zip(rest).zip(texts).map(
            |((mut meta, (audio, fingerprint)), text)| {
                meta.ingest_ms = summary.ms_per_track;
                Track {
                    meta,
                    audio,
                    text,
                    fingerprint,
                }
            },
        ));
    }

    keep.sort_by_key(|t| t.meta.id);
    summary.tracks = keep.len();
    Ok((keep, summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_keys() {
        for k in ["a/b.mp3", "x.OPUS", "tracks/3f2a9c1e-8b7d", "song.webm"] {
            assert!(is_audio(k), "{k}");
        }
        for k in [".DS_Store", "a/.hidden", "labels.tsv", "cover.jpg"] {
            assert!(!is_audio(k), "{k}");
        }
    }

    #[test]
    fn shards_partition_keys() {
        let keys: Vec<String> = (0..1000).map(|i| format!("{i}.mp3")).collect();
        let mut counts = [0; 4];
        for k in &keys {
            counts[shard_of(k, 4)] += 1;
        }
        assert_eq!(counts.iter().sum::<usize>(), 1000);
        assert!(counts.iter().all(|&c| c > 150), "{counts:?}");
        assert!(keys.iter().all(|k| shard_of(k, 1) == 0));
    }
}
