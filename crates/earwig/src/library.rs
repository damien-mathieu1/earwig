//! Snapshots and sharded indexing on top of a [`Store`].
//!
//! `earwig index` runs as one process or as N shard workers (e.g. a Kubernetes Indexed Job):
//! each worker analyses the media keys of its shard, reuses unchanged tracks of the current
//! snapshot, and writes `shards/<run>/<i>-of-<n>.bin`. The worker that finds all N shards
//! present merges them into `snapshots/<run>.bin` and moves `latest` to it. Merging is
//! deterministic, so two workers finishing at the same time publish the same snapshot.

use std::{collections::HashMap, time::Instant};

use anyhow::Context;
use earwig_analysis::{
    indexer::{self, Event, LocalFile, Source},
    Analyzer,
};
use earwig_core::{Index, TrackMeta};

use crate::storage::Store;

const KEEP_SNAPSHOTS: usize = 5;

/// The current snapshot: its key and content, or `None` if nothing was published yet.
pub async fn latest(store: &Store) -> anyhow::Result<Option<(String, Index)>> {
    let Some(pointer) = store.get_opt("latest").await? else {
        return Ok(None);
    };
    let key = String::from_utf8(pointer)?.trim().to_string();
    let index =
        Index::from_bytes(&store.get(&key).await?).with_context(|| format!("snapshot {key}"))?;
    Ok(Some((key, index)))
}

/// Current snapshot or a clear error.
pub async fn require_latest(store: &Store) -> anyhow::Result<Index> {
    latest(store).await?.map(|(_, i)| i).with_context(|| {
        format!(
            "no index in {} — run `earwig index <media>` first",
            store.url
        )
    })
}

/// Writes `snapshots/<run>.bin`, then points `latest` at it, then prunes old snapshots.
pub async fn publish(store: &Store, run: &str, index: &Index) -> anyhow::Result<String> {
    anyhow::ensure!(
        !index.tracks.is_empty(),
        "run {run}: no track could be analysed — the current snapshot is kept"
    );
    let key = format!("snapshots/{run}.bin");
    store.put(&key, index.to_bytes()?).await?;
    store.put("latest", key.clone().into_bytes()).await?;
    let mut snaps = store.list("snapshots").await?;
    snaps.sort_by_key(|e| std::cmp::Reverse(e.mtime));
    for old in snaps.iter().skip(KEEP_SNAPSHOTS).filter(|e| e.key != key) {
        store.delete(&old.key).await?;
    }
    Ok(key)
}

/// If the N shards of `run` are all present, merges and publishes them. Returns the snapshot key.
pub async fn merge(store: &Store, run: &str, shards: usize) -> anyhow::Result<Option<String>> {
    let prefix = format!("shards/{run}");
    let mut parts = store.list(&prefix).await?;
    parts.retain(|e| e.key.ends_with(&format!("-of-{shards}.bin")));
    if parts.len() < shards {
        return Ok(None);
    }
    parts.sort_by(|a, b| a.key.cmp(&b.key));
    let mut merged = Index::default();
    for p in &parts {
        let part = Index::from_bytes(&store.get(&p.key).await?).with_context(|| p.key.clone())?;
        anyhow::ensure!(
            merged.media.is_empty() || merged.media == part.media,
            "shards of run {run} index different media"
        );
        merged.media = part.media;
        merged.tracks.extend(part.tracks);
    }
    merged.tracks.sort_by_key(|t| t.meta.id);
    let key = publish(store, run, &merged).await?;
    for p in &parts {
        store.delete(&p.key).await?;
    }
    Ok(Some(key))
}

pub struct IndexJob {
    /// local directory or s3://bucket/prefix holding the audio files
    pub media: String,
    pub shard: usize,
    pub shards: usize,
    pub run: String,
}

/// Default run id: sortable UTC seconds.
pub fn new_run_id() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!("r{secs:012}")
}

/// One indexing worker. Returns the published snapshot key if this worker published one.
pub async fn index(
    store: &Store,
    job: &IndexJob,
    edit_meta: impl Fn(&mut TrackMeta) + Send + 'static,
) -> anyhow::Result<Option<String>> {
    anyhow::ensure!(
        job.shards >= 1 && job.shard < job.shards,
        "shard {} out of 0..{}",
        job.shard,
        job.shards
    );
    let media = Store::open(&job.media)?;
    let in_shard = |key: &str| indexer::shard_of(key, job.shards) == job.shard;

    // reuse the analysis of unchanged files from the current snapshot (same media only)
    let previous = match latest(store).await? {
        Some((_, idx)) if idx.media == media.url => idx
            .tracks
            .into_iter()
            .filter(|t| in_shard(&t.meta.path))
            .collect(),
        Some((key, idx)) => {
            eprintln!(
                "{key} indexes {} — starting from scratch for {}",
                idx.media, media.url
            );
            vec![]
        }
        None => vec![],
    };
    let audio: Vec<_> = media
        .list("")
        .await?
        .into_iter()
        .filter(|e| indexer::is_audio(&e.key))
        .collect();
    // a wrong path or an unreachable bucket must not publish an empty library over a good one
    anyhow::ensure!(
        !audio.is_empty(),
        "no audio file found in {} — nothing published",
        media.url
    );
    let sources: Vec<Source> = audio
        .into_iter()
        .filter(|e| in_shard(&e.key))
        .map(|e| Source {
            key: e.key,
            fingerprint: (e.size, e.mtime),
        })
        .collect();
    eprintln!(
        "shard {}/{}: {} media files, {} tracks in the current snapshot",
        job.shard,
        job.shards,
        sources.len(),
        previous.len()
    );

    let handle = tokio::runtime::Handle::current();
    let fetch_store = media.clone();
    let fetch = move |key: &str| -> anyhow::Result<LocalFile> {
        if fetch_store.is_local() {
            return Ok(LocalFile {
                path: std::path::Path::new(&fetch_store.url).join(key),
                temporary: false,
            });
        }
        // remote media: a temporary copy keeping the file name (title fallback, format probing)
        let bytes = handle.block_on(fetch_store.get(key))?;
        let dir = std::env::temp_dir().join(format!(
            "earwig-{}-{}",
            std::process::id(),
            earwig_core::track_id(key)
        ));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(
            std::path::Path::new(key)
                .file_name()
                .context("key without file name")?,
        );
        std::fs::write(&path, bytes)?;
        Ok(LocalFile {
            path,
            temporary: true,
        })
    };
    let t0 = Instant::now();
    let progress = move |e: Event| match e {
        Event::Started { files } => eprintln!(
            "models ready in {:.1}s — analysing {files} files",
            t0.elapsed().as_secs_f64()
        ),
        Event::Analysed { key, meta } => eprintln!(
            "  {:>6.1} bpm  {:<9}  {:<28}  {key}",
            meta.bpm,
            meta.key,
            meta.auto_genres.join(", ")
        ),
        Event::Skipped { key, error } => eprintln!("  skipped {key}: {error:#}"),
    };
    let (tracks, s) = tokio::task::spawn_blocking(move || {
        indexer::update(
            previous,
            &sources,
            fetch,
            Analyzer::load,
            edit_meta,
            progress,
        )
    })
    .await??;
    println!(
        "shard {}/{}: {} tracks ({} analysed at {:.0} ms/track, {} failed, {} removed)",
        job.shard, job.shards, s.tracks, s.analysed, s.ms_per_track, s.failed, s.removed
    );

    let part = Index {
        tracks,
        media: media.url.clone(),
    };
    if job.shards == 1 {
        return Ok(Some(publish(store, &job.run, &part).await?));
    }
    store
        .put(
            &format!("shards/{}/{:04}-of-{}.bin", job.run, job.shard, job.shards),
            part.to_bytes()?,
        )
        .await?;
    merge(store, &job.run, job.shards).await
}

/// Demo catalogue → metadata hook (files named <id>.wav).
pub fn catalog_editor(
    catalog: Vec<crate::catalog::CatalogTrack>,
) -> impl Fn(&mut TrackMeta) + Send + 'static {
    let by_id: HashMap<i64, crate::catalog::CatalogTrack> =
        catalog.into_iter().map(|t| (t.id, t)).collect();
    move |m: &mut TrackMeta| {
        let id = std::path::Path::new(&m.path)
            .file_stem()
            .and_then(|s| s.to_str()?.parse::<i64>().ok());
        if let Some(c) = id.and_then(|id| by_id.get(&id)) {
            (m.title, m.artist, m.genre, m.subgenre, m.year) = (
                c.title.clone(),
                c.artist.clone(),
                c.genre.clone(),
                c.subgenre.clone(),
                Some(c.year),
            );
            (m.moods, m.themes, m.tags) = (c.moods.clone(), c.themes.clone(), c.tags.clone());
            (m.bpm_label, m.key_label) = (Some(c.bpm), Some(c.key.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use earwig_core::{Track, TrackMeta};

    fn track(key: &str) -> Track {
        Track {
            meta: TrackMeta {
                id: earwig_core::track_id(key),
                path: key.into(),
                ..Default::default()
            },
            audio: vec![1.0],
            text: vec![1.0],
            fingerprint: (1, 1),
        }
    }

    #[tokio::test]
    async fn last_shard_merges_and_publishes() {
        let dir = std::env::temp_dir().join(format!("earwig-lib-test-{}", std::process::id()));
        let store = Store::open(dir.to_str().unwrap()).unwrap();
        let part = |keys: &[&str]| Index {
            tracks: keys.iter().map(|k| track(k)).collect(),
            media: "m".into(),
        };
        store
            .put(
                "shards/r1/0000-of-2.bin",
                part(&["a.mp3", "b.mp3"]).to_bytes().unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            merge(&store, "r1", 2).await.unwrap(),
            None,
            "one shard missing"
        );
        store
            .put(
                "shards/r1/0001-of-2.bin",
                part(&["c.mp3"]).to_bytes().unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            merge(&store, "r1", 2).await.unwrap().as_deref(),
            Some("snapshots/r1.bin")
        );
        let (key, idx) = latest(&store).await.unwrap().unwrap();
        assert_eq!((key.as_str(), idx.tracks.len()), ("snapshots/r1.bin", 3));
        assert!(
            store.list("shards/r1").await.unwrap().is_empty(),
            "shards cleaned up"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
