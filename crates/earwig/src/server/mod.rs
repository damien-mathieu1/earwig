//! `earwig serve`: a stateless replica. It loads the current snapshot from the store into RAM
//! (scoring is a brute-force scan per request), polls `latest` and swaps in new snapshots
//! without downtime, and keeps nothing on its local disk: ratings and audio live in the store
//! too, so any number of replicas can run behind a load balancer.
//!
//! Two APIs share one search engine ([`run_search`]):
//! * `/api/v1/*` ([`v1`]) — the stable, documented API for other services (keys, several seeds,
//!   exclusions), optionally behind a bearer token;
//! * `/api/*` — what the bundled web UI uses (benchmark, ratings…); off with `--api-only`.
//!
//! The analysis models are only loaded on the first upload, the CLAP text tower on the first
//! search by mood / style / description.

mod v1;

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

use arc_swap::ArcSwapOption;
use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Path, Query as UrlQuery, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use earwig_analysis::{Analyzer, ClapVocab, GENRES, MOODS};
use earwig_core::{
    scoring::{self, bpm_delta, cosine, mix_top_k_by, Scorable, SubScores, Weights},
    Index, Query, Track, TrackMeta,
};
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use anyhow::Context;

use crate::{library, storage::Store};

const FRONTEND: &str = include_str!("../../frontend/index.html");

/// The library being served: the current snapshot plus the tracks pushed through
/// `PUT /api/v1/tracks/{id}` (the overlay). Replaced as a whole when either changes.
struct Lib {
    index: Index,
    by_id: HashMap<i64, usize>,
    by_key: HashMap<String, usize>,
    /// snapshot key, or "(no snapshot)" when every track was pushed through the API
    key: String,
    /// where the audio files live (none for an overlay-only library without `--media`)
    media: Option<Store>,
    load_ms: f64,
}

/// Version of a stored object: (size, mtime, etag).
type Version = (u64, u64, Option<String>);

/// What the current [`Lib`] was built from.
#[derive(Default)]
struct Sources {
    snapshot: Option<String>,
    /// overlay object key -> version
    overlay: HashMap<String, Version>,
}

/// Tracks pushed through the API: one object per track, `overlay/<hex of the id>`.
const OVERLAY: &str = "overlay/";

fn overlay_key(id: &str) -> String {
    let hex: String = id.bytes().map(|b| format!("{b:02x}")).collect();
    format!("{OVERLAY}{hex}")
}

fn overlay_id(key: &str) -> Option<String> {
    let hex = key.strip_prefix(OVERLAY)?.as_bytes();
    let bytes = hex
        .chunks(2)
        .map(|c| u8::from_str_radix(std::str::from_utf8(c).ok()?, 16).ok())
        .collect::<Option<Vec<u8>>>()?;
    String::from_utf8(bytes).ok()
}

struct App {
    store: Store,
    lib: ArcSwapOption<Lib>,
    sources: tokio::sync::Mutex<Sources>,
    /// `--media`: overrides the snapshot's media (and gives one to an overlay-only library)
    media_url: Option<String>,
    /// analyses running at once on this replica (each takes ~1 s of CPU and some RAM)
    analysis: tokio::sync::Semaphore,
    /// contents of the MTG-Jamendo style TSVs given with `--labels` (empty = index fields)
    label_tsvs: Vec<String>,
    analyzer: OnceCell<Arc<Analyzer>>,
    vocab: OnceCell<Arc<ClapVocab>>,
}

impl App {
    /// The current snapshot, or 503 while none is loaded (fresh store, store unreachable at boot).
    fn lib(&self) -> Result<Arc<Lib>, ApiError> {
        self.lib.load_full().ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "no index loaded yet".into(),
        ))
    }

    async fn vocab(&self) -> Result<Arc<ClapVocab>, ApiError> {
        Ok(self
            .vocab
            .get_or_try_init(|| async {
                tokio::task::spawn_blocking(ClapVocab::load)
                    .await
                    .map_err(internal)?
                    .map(Arc::new)
                    .map_err(internal)
            })
            .await?
            .clone())
    }

    async fn analyzer(&self) -> Result<Arc<Analyzer>, ApiError> {
        Ok(self
            .analyzer
            .get_or_try_init(|| async {
                tokio::task::spawn_blocking(Analyzer::load)
                    .await
                    .map_err(internal)?
                    .map(Arc::new)
                    .map_err(internal)
            })
            .await?
            .clone())
    }
}

type AppState = Arc<App>;
type ApiError = (StatusCode, String);

fn internal(e: impl std::fmt::Display) -> ApiError {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

fn bad_request(m: impl Into<String>) -> ApiError {
    (StatusCode::BAD_REQUEST, m.into())
}

// ───────────────────────────── shared search engine ─────────────────────────────

/// Everything a search can combine, whatever API it came through.
#[derive(Default)]
struct SearchSpec {
    /// the reference sound + tags (one track, an upload, or the centroid of several seeds)
    reference: Option<Query>,
    /// library positions never returned (the seeds, already-heard tracks…)
    exclude: HashSet<usize>,
    /// when set, only these library positions may be returned (the app's visibility rules)
    allowed: Option<HashSet<usize>>,
    /// sound + tags to move away from (skipped / disliked tracks)
    avoid: Option<Query>,
    /// at most this many results per artist / per album (empty names are not capped)
    max_per_artist: Option<usize>,
    max_per_album: Option<usize>,
    moods: Vec<String>,
    genres: Vec<String>,
    description: String,
    /// target tempo; defaults to the reference's
    bpm: Option<f32>,
    /// track (reference sound), tags, mood, genre, description, bpm, avoid
    weights: [f32; 7],
    bpm_tolerance: f32,
    k: usize,
}

struct Hit {
    index: usize,
    score: f32,
    /// raw (not standardised) value of every part that took part
    parts: Vec<(&'static str, f32)>,
    /// the part that added most to the score (largest weighted standardised value)
    driven_by: &'static str,
}

/// Mean of the label vectors (vocabulary entries are pre-computed, other words embedded on the fly).
fn labels_vector(
    vocab: &ClapVocab,
    labels: &[String],
    known: &[&str],
    vecs: &[Vec<f32>],
    kind: &str,
) -> anyhow::Result<Option<Vec<f32>>> {
    let mut sum = vec![0f32; 512];
    let mut n = 0;
    for l in labels.iter().map(|l| l.trim()).filter(|l| !l.is_empty()) {
        let v = match known.iter().position(|k| k.eq_ignore_ascii_case(l)) {
            Some(i) => vecs[i].clone(),
            None => vocab.embed_label(l, kind)?,
        };
        sum.iter_mut().zip(v).for_each(|(s, x)| *s += x);
        n += 1;
    }
    if n == 0 {
        return Ok(None);
    }
    earwig_core::l2_normalize(&mut sum);
    Ok(Some(sum))
}

/// Several seeds → one profile: mean of the (normalised) sound and tag vectors, median tempo.
/// Weighted mean of normalised seed vectors; tempo = median of the seeds that count.
fn centroid(seeds: &[(&dyn Scorable, f32)]) -> Option<Query> {
    let seeds: Vec<_> = seeds.iter().filter(|(_, w)| *w > 0.0).collect();
    let (first, _) = seeds.first()?;
    let mut audio = vec![0f32; first.audio().len()];
    let mut text = vec![0f32; first.text().len()];
    for (s, w) in &seeds {
        audio
            .iter_mut()
            .zip(s.audio())
            .for_each(|(a, x)| *a += w * x);
        text.iter_mut().zip(s.text()).for_each(|(a, x)| *a += w * x);
    }
    earwig_core::l2_normalize(&mut audio);
    earwig_core::l2_normalize(&mut text);
    let mut bpms: Vec<f32> = seeds.iter().map(|(s, _)| s.bpm()).collect();
    bpms.sort_by(f32::total_cmp);
    Some(Query {
        audio,
        text,
        bpm: bpms[bpms.len() / 2],
    })
}

/// Mix of reference (sound + tags), moods, styles, free description and tempo; each part is
/// standardised over the library before weighting (`scoring::mix_top_k`).
async fn run_search(app: &App, lib: &Lib, spec: SearchSpec) -> Result<(Vec<Hit>, f64), ApiError> {
    let [w_track, w_tags, w_mood, w_genre, w_description, w_bpm, w_avoid] =
        spec.weights.map(|w| w.max(0.0));
    for q in [&spec.reference, &spec.avoid].into_iter().flatten() {
        let t0 = &lib.index.tracks[0];
        if q.audio.len() != t0.audio.len() || q.text.len() != t0.text.len() {
            return Err(bad_request("query vectors have the wrong dimensions"));
        }
    }

    // words -> CLAP text vectors (text tower loaded on first use)
    let wants_words = (w_mood > 0.0 && !spec.moods.is_empty())
        || (w_genre > 0.0 && !spec.genres.is_empty())
        || (w_description > 0.0 && !spec.description.trim().is_empty());
    let (mut mood, mut genre, mut description) = (None, None, None);
    if wants_words {
        let vocab = app.vocab().await?;
        let (moods, genres, desc) = (
            spec.moods.clone(),
            spec.genres.clone(),
            spec.description.clone(),
        );
        (mood, genre, description) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            Ok((
                labels_vector(&vocab, &moods, MOODS, &vocab.moods, "mood")?,
                labels_vector(&vocab, &genres, GENRES, &vocab.genres, "genre")?,
                labels_vector(&vocab, &[desc], &[], &[], "mood")?,
            ))
        })
        .await
        .map_err(internal)?
        .map_err(internal)?;
    }

    let t0 = Instant::now();
    let tracks = &lib.index.tracks;
    let per_track = |f: &dyn Fn(&Track) -> f32| tracks.iter().map(f).collect::<Vec<f32>>();
    let mut parts: Vec<(&'static str, f32, Vec<f32>)> = Vec::new();
    if let Some(r) = &spec.reference {
        if w_track > 0.0 {
            parts.push(("track", w_track, per_track(&|t| cosine(&r.audio, &t.audio))));
        }
        if w_tags > 0.0 {
            parts.push(("tags", w_tags, per_track(&|t| cosine(&r.text, &t.text))));
        }
    }
    for (name, w, v) in [
        ("mood", w_mood, &mood),
        ("genre", w_genre, &genre),
        ("description", w_description, &description),
    ] {
        if let (true, Some(v)) = (w > 0.0, v) {
            parts.push((name, w, per_track(&|t| cosine(v, &t.audio))));
        }
    }
    if let (true, Some(a)) = (w_avoid > 0.0, &spec.avoid) {
        // minus the similarity (sound and tags) to the avoided tracks
        parts.push((
            "avoid",
            w_avoid,
            per_track(&|t| -(cosine(&a.audio, &t.audio) + cosine(&a.text, &t.text)) / 2.0),
        ));
    }
    if let Some(target) = spec
        .bpm
        .or(spec.reference.as_ref().map(|r| r.bpm))
        .filter(|_| w_bpm > 0.0)
    {
        let tol = spec.bpm_tolerance.max(1e-3);
        parts.push((
            "bpm",
            w_bpm,
            per_track(&|t| (1.0 - bpm_delta(target, t.meta.bpm) / tol).max(0.0)),
        ));
    }
    if parts.is_empty() {
        return Err(bad_request("nothing to recommend from: give seeds, a mood, a style, a description or a tempo, with a non-zero weight"));
    }
    let components: Vec<(f32, Vec<f32>)> = parts.iter().map(|(_, w, v)| (*w, v.clone())).collect();
    let name_of = |s: &str| s.trim().to_lowercase();
    let mut per_artist: HashMap<String, usize> = HashMap::new();
    let mut per_album: HashMap<(String, String), usize> = HashMap::new();
    let mut under_cap = |i: usize| {
        let m = &tracks[i].meta;
        let (artist, album) = (name_of(&m.artist), name_of(&m.album));
        let artist_full = spec.max_per_artist.is_some_and(|max| {
            !artist.is_empty() && per_artist.get(&artist).copied().unwrap_or(0) >= max
        });
        let album_key = (artist.clone(), album);
        let album_full = spec.max_per_album.is_some_and(|max| {
            !album_key.1.is_empty() && per_album.get(&album_key).copied().unwrap_or(0) >= max
        });
        if artist_full || album_full {
            return false;
        }
        *per_artist.entry(artist).or_default() += 1;
        *per_album.entry(album_key).or_default() += 1;
        true
    };
    let top = mix_top_k_by(
        &components,
        |i| spec.exclude.contains(&i) || spec.allowed.as_ref().is_some_and(|a| !a.contains(&i)),
        spec.k.clamp(1, 100),
        &mut under_cap,
    );
    // mean / std of each part, as mix_top_k standardises them, to tell which one led
    let stats: Vec<(f32, f32)> = parts
        .iter()
        .map(|(_, _, v)| {
            let n = v.len().max(1) as f32;
            let mean = v.iter().sum::<f32>() / n;
            let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
            (mean, std.max(1e-6))
        })
        .collect();
    let hits = top
        .into_iter()
        .map(|(index, score)| Hit {
            index,
            score,
            parts: parts.iter().map(|(n, _, v)| (*n, v[index])).collect(),
            driven_by: parts
                .iter()
                .zip(&stats)
                .map(|((n, w, v), (mean, std))| (*n, w * (v[index] - mean) / std))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map_or("", |(n, _)| n),
        })
        .collect();
    let secs = t0.elapsed().as_secs_f64();
    metrics::histogram!("earwig_search_duration_seconds").record(secs);
    Ok((hits, secs * 1e3))
}

/// Writes an uploaded file to a temp file, analyses it, returns its metadata and features.
/// Writes audio bytes to a temp file and analyses it (sound, tempo, tags): the
/// `(metadata, audio vector)` of `Analyzer::analyze`, title falling back to `name`'s stem.
async fn analyze_bytes(
    app: &App,
    name: &str,
    body: Bytes,
) -> Result<(TrackMeta, Vec<f32>), ApiError> {
    let analyzer = app.analyzer().await?;
    let _permit = app.analysis.acquire().await.map_err(internal)?;
    // symphonia/lofty probe the content; a short alphanumeric extension is only a hint
    let ext: String = name
        .rsplit_once('.')
        .map(|(_, e)| e.to_ascii_lowercase())
        .filter(|e| e.len() <= 5 && e.chars().all(|c| c.is_ascii_alphanumeric()))
        .unwrap_or_else(|| "bin".into());
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let tmp = std::env::temp_dir().join(format!(
        "earwig-upload-{}-{nanos}.{ext}",
        std::process::id()
    ));
    tokio::fs::write(&tmp, &body).await.map_err(internal)?;
    let t0 = Instant::now();
    let path = tmp.clone();
    let res = tokio::task::spawn_blocking(move || analyzer.analyze(&path)).await;
    let _ = tokio::fs::remove_file(&tmp).await;
    let ok = matches!(res, Ok(Ok(_)));
    metrics::counter!("earwig_analyses_total", "result" => if ok { "ok" } else { "error" })
        .increment(1);
    metrics::histogram!("earwig_analysis_duration_seconds").record(t0.elapsed().as_secs_f64());
    let (mut meta, audio) = res
        .map_err(internal)?
        .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, format!("{e:#}")))?;
    if meta.title.starts_with("earwig-upload-") {
        // no title tag: fall back to the client's file name instead of our temp name
        let stem = name.rsplit('/').next().unwrap_or(name);
        meta.title = stem
            .rsplit_once('.')
            .map_or(stem, |(stem, _)| stem)
            .to_string();
    }
    meta.ingest_ms = (t0.elapsed().as_secs_f64() * 1e3) as f32;
    Ok((meta, audio))
}

/// An uploaded file's metadata and features (text vector from its own tags).
async fn analyze_upload(
    app: &App,
    name: Option<String>,
    body: Bytes,
) -> Result<(TrackMeta, Query), ApiError> {
    let name = name.unwrap_or_default();
    let (mut meta, audio) = analyze_bytes(app, &name, body).await?;
    let text = embed_text(app, &meta).await?;
    meta.path = name;
    meta.id = -1;
    let bpm = meta.bpm;
    Ok((meta, Query { audio, text, bpm }))
}

/// MiniLM embedding of the track's text document.
async fn embed_text(app: &App, meta: &TrackMeta) -> Result<Vec<f32>, ApiError> {
    let analyzer = app.analyzer().await?;
    let doc = meta.text_document();
    tokio::task::spawn_blocking(move || analyzer.embed_documents(&[doc]))
        .await
        .map_err(internal)?
        .map(|mut v| v.remove(0))
        .map_err(internal)
}

// ───────────────────────────── /api (web UI) ─────────────────────────────

async fn tracks(State(s): State<AppState>) -> Result<Json<Vec<TrackMeta>>, ApiError> {
    Ok(Json(
        s.lib()?
            .index
            .tracks
            .iter()
            .map(|t| t.meta.clone())
            .collect(),
    ))
}

#[derive(Deserialize)]
struct RecoRequest {
    /// a track of the index…
    track_id: Option<i64>,
    /// …or the features of an uploaded file (from `/api/analyze`)
    query: Option<Query>,
    #[serde(flatten)]
    weights: Weights,
    #[serde(default = "default_k")]
    k: usize,
}
fn default_k() -> usize {
    10
}

#[derive(Serialize)]
struct RecoItem<'a> {
    track: &'a TrackMeta,
    #[serde(flatten)]
    scores: SubScores,
}

#[derive(Serialize)]
struct RecoResponse<'a> {
    source: Option<&'a TrackMeta>,
    results: Vec<RecoItem<'a>>,
    query_ms: f64,
}

fn dims_ok(lib: &Lib, q: &Query) -> bool {
    q.audio.len() == lib.index.tracks[0].audio.len()
        && q.text.len() == lib.index.tracks[0].text.len()
}

async fn recommend(
    State(s): State<AppState>,
    Json(req): Json<RecoRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let lib = s.lib()?;
    let tracks = &lib.index.tracks;
    let k = req.k.min(100);
    let t0 = Instant::now();
    let (source, top) = match (req.track_id, &req.query) {
        (Some(id), _) => {
            let &i = lib
                .by_id
                .get(&id)
                .ok_or((StatusCode::NOT_FOUND, format!("no track {id}")))?;
            (
                Some(&tracks[i].meta),
                scoring::top_k(tracks, &tracks[i], Some(i), &req.weights, k),
            )
        }
        (None, Some(q)) if dims_ok(&lib, q) => {
            (None, scoring::top_k(tracks, q, None, &req.weights, k))
        }
        (None, Some(_)) => return Err(bad_request("query vectors have the wrong dimensions")),
        (None, None) => return Err(bad_request("track_id or query required")),
    };
    let query_ms = t0.elapsed().as_secs_f64() * 1e3;
    let results = top
        .into_iter()
        .map(|(i, scores)| RecoItem {
            track: &tracks[i].meta,
            scores,
        })
        .collect();
    serde_json::to_value(RecoResponse {
        source,
        results,
        query_ms,
    })
    .map(Json)
    .map_err(internal)
}

#[derive(Deserialize)]
struct AnalyzeParams {
    name: Option<String>,
}

/// Body = raw audio file. Returns its metadata + query features for `/api/search`.
async fn analyze(
    State(s): State<AppState>,
    UrlQuery(p): UrlQuery<AnalyzeParams>,
    body: Bytes,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (meta, query) = analyze_upload(&s, p.name, body).await?;
    Ok(Json(serde_json::json!({ "meta": meta, "query": query })))
}

#[derive(Deserialize)]
struct SearchRequest {
    track_id: Option<i64>,
    query: Option<Query>,
    #[serde(default)]
    moods: Vec<String>,
    #[serde(default)]
    genres: Vec<String>,
    #[serde(default)]
    description: String,
    bpm: Option<f32>,
    #[serde(default)]
    w_track: f32,
    #[serde(default)]
    w_tags: f32,
    #[serde(default)]
    w_mood: f32,
    #[serde(default)]
    w_genre: f32,
    #[serde(default)]
    w_description: f32,
    #[serde(default)]
    w_bpm: f32,
    #[serde(default = "default_tolerance")]
    bpm_tolerance: f32,
    #[serde(default = "default_k")]
    k: usize,
}
fn default_tolerance() -> f32 {
    20.0
}

async fn search(
    State(s): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let lib = s.lib()?;
    let tracks = &lib.index.tracks;
    let (source, reference, exclude) = match (req.track_id, req.query) {
        (Some(id), _) => {
            let &i = lib
                .by_id
                .get(&id)
                .ok_or((StatusCode::NOT_FOUND, format!("no track {id}")))?;
            let t = &tracks[i];
            (
                Some(&t.meta),
                Some(Query {
                    audio: t.audio.clone(),
                    text: t.text.clone(),
                    bpm: t.meta.bpm,
                }),
                HashSet::from([i]),
            )
        }
        (None, q) => (None, q, HashSet::new()),
    };
    let spec = SearchSpec {
        reference,
        exclude,
        moods: req.moods,
        genres: req.genres,
        description: req.description,
        bpm: req.bpm,
        weights: [
            req.w_track,
            req.w_tags,
            req.w_mood,
            req.w_genre,
            req.w_description,
            req.w_bpm,
            0.0,
        ],
        bpm_tolerance: req.bpm_tolerance,
        k: req.k,
        ..Default::default()
    };
    let (hits, query_ms) = run_search(&s, &lib, spec).await?;
    let results: Vec<_> = hits
        .iter()
        .map(|h| {
            let parts: serde_json::Map<String, serde_json::Value> = h
                .parts
                .iter()
                .map(|(n, v)| (n.to_string(), (*v).into()))
                .collect();
            serde_json::json!({ "track": tracks[h.index].meta, "score": h.score, "parts": parts })
        })
        .collect();
    Ok(Json(
        serde_json::json!({ "source": source, "results": results, "query_ms": query_ms }),
    ))
}

/// The original file, read from the media store, so results can be listened to.
async fn audio(State(s): State<AppState>, Path(id): Path<i64>) -> Result<Response, ApiError> {
    let lib = s.lib()?;
    let &i = lib
        .by_id
        .get(&id)
        .ok_or((StatusCode::NOT_FOUND, String::new()))?;
    let key = &lib.index.tracks[i].meta.path;
    let bytes = lib
        .media
        .as_ref()
        .ok_or((StatusCode::NOT_FOUND, "no media store".into()))?
        .get(key)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, format!("{e:#}")))?;
    let mime = match std::path::Path::new(key)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        "wav" => "audio/wav",
        "ogg" | "oga" => "audio/ogg",
        "m4a" | "mp4" | "aac" => "audio/mp4",
        "aif" | "aiff" => "audio/aiff",
        _ => "application/octet-stream",
    };
    Ok(([(header::CONTENT_TYPE, mime)], Body::from(bytes)).into_response())
}

/// Resident set size of this process (`/proc` on Linux, `ps` elsewhere).
fn rss_mb() -> f64 {
    // Linux (incl. slim containers without `ps`): /proc/self/status "VmRSS:   12345 kB"
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        let kb = status.lines().find_map(|l| {
            l.strip_prefix("VmRSS:")?
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<f64>()
                .ok()
        });
        return kb.map_or(0.0, |kb| kb / 1024.0);
    }
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<f64>()
                .ok()
        })
        .map_or(0.0, |kb| kb / 1024.0)
}

async fn stats(State(s): State<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    let lib = s.lib()?;
    let t = &lib.index.tracks;
    Ok(Json(serde_json::json!({
        "engine": "earwig · symphonia + stratum-dsp + CLAP/ort + fastembed → bincode → axum",
        "tracks": t.len(),
        "snapshot": lib.key,
        "replica": std::env::var("HOSTNAME").unwrap_or_default(),
        "avg_ingest_ms": t.iter().map(|t| t.meta.ingest_ms as f64).sum::<f64>() / t.len().max(1) as f64,
        "index_load_ms": lib.load_ms,
        "rss_mb": rss_mb(),
        "audio_dim": t.first().map_or(0, |t| t.audio.len()),
        "text_dim": t.first().map_or(0, |t| t.text.len()),
        // ground truth available to the Benchmark tab
        "eval_by": if s.label_tsvs.is_empty() { vec!["artist", "genre", "subgenre"] } else { vec!["genre", "artist", "mood"] },
        "labels_from": if s.label_tsvs.is_empty() { "index" } else { "tsv" },
    })))
}

#[derive(Deserialize)]
struct EvalRequest {
    by: String,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default = "default_tolerance")]
    bpm_tolerance: f32,
    #[serde(default)]
    grid: bool,
}

/// Benchmark tab: same report as `earwig eval` (can take a few seconds on big libraries).
async fn eval(
    State(s): State<AppState>,
    Json(req): Json<EvalRequest>,
) -> Result<Json<crate::eval::Report>, ApiError> {
    let lib = s.lib()?;
    let k = req.k.clamp(1, 100);
    tokio::task::spawn_blocking(move || {
        let labels = crate::eval::track_labels(&lib.index.tracks, &req.by, &s.label_tsvs)?;
        crate::eval::evaluate(
            &lib.index.tracks,
            &labels,
            &req.by,
            k,
            req.bpm_tolerance,
            req.grid,
        )
    })
    .await
    .map_err(internal)?
    .map(Json)
    .map_err(|e| bad_request(format!("{e:#}")))
}

#[derive(Deserialize)]
struct LabelsParams {
    by: String,
}

/// Ground truth per track id, to show next to results.
async fn labels(
    State(s): State<AppState>,
    UrlQuery(p): UrlQuery<LabelsParams>,
) -> Result<Json<HashMap<i64, Vec<String>>>, ApiError> {
    let lib = s.lib()?;
    let labels = crate::eval::track_labels(&lib.index.tracks, &p.by, &s.label_tsvs)
        .map_err(|e| bad_request(format!("{e:#}")))?;
    Ok(Json(
        lib.index
            .tracks
            .iter()
            .zip(labels)
            .filter(|(_, l)| !l.is_empty())
            .map(|(t, l)| (t.meta.id, l.into_iter().collect()))
            .collect(),
    ))
}

#[derive(Deserialize, Serialize)]
struct Rating {
    source_id: i64,
    track_id: i64,
    relevant: bool,
    /// the weightings whose top-k contained this track (blind A/B test)
    configs: Vec<String>,
    #[serde(default)]
    at: u64,
}

/// One object per rating: replicas never contend on a shared file.
async fn add_rating(
    State(s): State<AppState>,
    Json(mut r): Json<Rating>,
) -> Result<StatusCode, ApiError> {
    let lib = s.lib()?;
    if !lib.by_id.contains_key(&r.source_id) || !lib.by_id.contains_key(&r.track_id) {
        return Err(bad_request("unknown track"));
    }
    if r.configs.is_empty()
        || r.configs.len() > 4
        || r.configs.iter().any(|c| c.is_empty() || c.len() > 64)
    {
        return Err(bad_request(
            "configs: 1 to 4 names of at most 64 characters",
        ));
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    r.at = now.as_secs();
    let key = format!("ratings/{:020}-{}.json", now.as_nanos(), unique_suffix());
    s.store
        .put(&key, serde_json::to_vec(&r).map_err(internal)?)
        .await
        .map_err(internal)?;
    Ok(StatusCode::NO_CONTENT)
}

/// Uniqueness within a nanosecond across replicas: process id + a per-process counter.
fn unique_suffix() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    format!(
        "{}-{}",
        std::process::id(),
        N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

async fn list_ratings(State(s): State<AppState>) -> Result<Json<Vec<Rating>>, ApiError> {
    let entries = s.store.list("ratings").await.map_err(internal)?;
    let bodies = futures::future::try_join_all(entries.iter().map(|e| s.store.get(&e.key)))
        .await
        .map_err(internal)?;
    // an unreadable object is skipped, not fatal
    Ok(Json(
        bodies
            .iter()
            .filter_map(|b| serde_json::from_slice(b).ok())
            .collect(),
    ))
}

// ───────────────────────────── snapshots, auth, router ─────────────────────────────

fn load_lib(key: String, index: Index) -> anyhow::Result<Lib> {
    let by_id = index
        .tracks
        .iter()
        .enumerate()
        .map(|(i, t)| (t.meta.id, i))
        .collect();
    let by_key = index
        .tracks
        .iter()
        .enumerate()
        .map(|(i, t)| (t.meta.path.clone(), i))
        .collect();
    // recommendations never read the audio: an unreachable media store (a folder that exists
    // only on the indexing machine, a container without that mount…) only disables playback
    // and `media_key` pushes, it must not keep the library from loading
    let media = (!index.media.is_empty())
        .then(|| Store::open(&index.media))
        .and_then(|r| {
            r.map_err(|e| tracing::warn!("media {}: {e:#} (audio playback disabled)", index.media))
                .ok()
        });
    Ok(Lib {
        index,
        by_id,
        by_key,
        key,
        media,
        load_ms: 0.0,
    })
}

/// Rebuilds the library when the snapshot or the overlay changed; keeps serving the old one if
/// anything fails.
async fn refresh(s: &App) {
    if let Err(e) = try_refresh(s).await {
        metrics::counter!("earwig_refresh_errors_total").increment(1);
        tracing::warn!("refresh: {e:#}");
    }
}

async fn try_refresh(s: &App) -> anyhow::Result<()> {
    let mut src = s.sources.lock().await;
    let pointer = s
        .store
        .get_opt("latest")
        .await?
        .map(|p| String::from_utf8_lossy(&p).trim().to_string());
    let listed: HashMap<String, Version> = s
        .store
        .list(OVERLAY)
        .await?
        .into_iter()
        .map(|e| (e.key, (e.size, e.mtime, e.etag)))
        .collect();
    let changed: Vec<&String> = listed
        .iter()
        .filter(|(k, v)| src.overlay.get(*k) != Some(*v))
        .map(|(k, _)| k)
        .collect();
    let removed = src.overlay.keys().any(|k| !listed.contains_key(k));
    let new_snapshot = pointer.is_some() && pointer != src.snapshot;
    if !new_snapshot && !removed && changed.is_empty() {
        return Ok(());
    }

    let t0 = Instant::now();
    let mut fresh = Vec::new();
    let mut failed = HashSet::new();
    for &k in &changed {
        match s.store.get(k).await.and_then(|b| Index::from_bytes(&b)) {
            Ok(i) => fresh.extend(i.tracks),
            Err(e) => {
                tracing::warn!("refresh: skipping {k}: {e:#}");
                failed.insert(k.clone());
            }
        }
    }
    let fresh_paths: HashSet<String> = fresh.iter().map(|t| t.meta.path.clone()).collect();
    let current = s.lib.load_full();
    // shortcut: rebuilding clones the track vector (2× RAM for a moment), fine up to ~1M
    // tracks; past that, fold the overlay into the snapshots instead
    let (snapshot, mut tracks, media) = if let (Some(l), false) =
        (&current, new_snapshot || removed)
    {
        let tracks = l
            .index
            .tracks
            .iter()
            .filter(|t| !fresh_paths.contains(&t.meta.path))
            .cloned()
            .collect();
        (src.snapshot.clone(), tracks, l.index.media.clone())
    } else {
        let overlay_paths: HashSet<String> = listed.keys().filter_map(|k| overlay_id(k)).collect();
        let (snapshot, base, media) = match &pointer {
            Some(_) => {
                let (key, index) = library::latest(&s.store)
                    .await?
                    .context("the latest pointer names no snapshot")?;
                anyhow::ensure!(
                    !index.tracks.is_empty(),
                    "{key} is empty, keeping the current library"
                );
                (Some(key), index.tracks, index.media)
            }
            None => (None, Vec::new(), String::new()),
        };
        let mut tracks: Vec<Track> = base
            .into_iter()
            .filter(|t| !overlay_paths.contains(&t.meta.path))
            .collect();
        // overlay tracks that did not change are already in memory
        if let Some(l) = &current {
            tracks.extend(
                l.index
                    .tracks
                    .iter()
                    .filter(|t| {
                        overlay_paths.contains(&t.meta.path) && !fresh_paths.contains(&t.meta.path)
                    })
                    .cloned(),
            );
        }
        (snapshot, tracks, media)
    };
    tracks.extend(fresh);
    // an explicit --media wins over the snapshot's (media moved since indexing, other mount…)
    let media = s.media_url.clone().unwrap_or(media);

    let key = snapshot.clone().unwrap_or_else(|| "(no snapshot)".into());
    let overlay = listed.len() - failed.len();
    if tracks.is_empty() {
        s.lib.store(None);
    } else {
        let mut lib = load_lib(key.clone(), Index { tracks, media })?;
        lib.load_ms = t0.elapsed().as_secs_f64() * 1e3;
        tracing::info!(
            snapshot = %key,
            pushed = overlay,
            tracks = lib.index.tracks.len(),
            ms = lib.load_ms,
            "library loaded"
        );
        metrics::gauge!("earwig_tracks").set(lib.index.tracks.len() as f64);
        metrics::gauge!("earwig_tracks_pushed").set(overlay as f64);
        metrics::gauge!("earwig_library_load_seconds").set(lib.load_ms / 1e3);
        s.lib.store(Some(Arc::new(lib)));
    }
    src.snapshot = snapshot;
    src.overlay = listed
        .into_iter()
        .filter(|(k, _)| !failed.contains(k))
        .collect();
    Ok(())
}

/// Constant-time comparison, so the token cannot be guessed from response times.
fn same_secret(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// With `--api-token`, every `/api/*` route requires `Authorization: Bearer <token>`. Any of
/// the tokens is accepted, so a new one can be rolled out before the old one is removed.
async fn require_token(
    State(tokens): State<Arc<Vec<String>>>,
    req: Request,
    next: Next,
) -> Response {
    let ok = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|t| {
            // check them all: the time taken does not tell which one matched
            tokens.iter().fold(false, |hit, k| {
                same_secret(t.as_bytes(), k.as_bytes()) | hit
            })
        });
    if ok {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            Json(serde_json::json!({ "error": "missing or invalid bearer token" })),
        )
            .into_response()
    }
}

/// What a replica does, so analysis (CPU-heavy) never slows down recommendations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum Role {
    /// everything (one process, small setups)
    #[default]
    All,
    /// recommendations and lookups only: never loads the analysis models
    Api,
    /// receives pushed tracks (`PUT` / `DELETE /api/v1/tracks`) and analyses files
    Indexer,
}

/// More requests than this in flight on `/api`: 503 with `Retry-After`, instead of queueing.
async fn shed_load(
    State(slots): State<Arc<tokio::sync::Semaphore>>,
    req: Request,
    next: Next,
) -> Response {
    match slots.try_acquire() {
        Ok(_permit) => next.run(req).await,
        Err(_) => {
            metrics::counter!("earwig_http_shed_total").increment(1);
            (
                StatusCode::SERVICE_UNAVAILABLE,
                [(header::RETRY_AFTER, "1")],
                Json(serde_json::json!({ "error": "overloaded, retry later" })),
            )
                .into_response()
        }
    }
}

pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub label_tsvs: Vec<String>,
    pub refresh_secs: u64,
    /// bearer token(s) required on `/api/*`, comma-separated (health probes stay open)
    pub api_token: Option<String>,
    pub role: Role,
    /// requests in flight on `/api` before shedding load
    pub max_concurrent: usize,
    /// only `/api/v1` and the health probes: no web UI, no benchmark, ratings or legacy routes
    pub api_only: bool,
    /// media store, overriding the snapshot's (audio playback, pushed tracks' `media_key`)
    pub media: Option<String>,
}

/// The routes, with the current snapshot loaded (used by `serve` and by the tests).
async fn router(store: Store, cfg: &ServeConfig) -> (Router, Arc<App>) {
    let app = Arc::new(App {
        store,
        lib: ArcSwapOption::empty(),
        sources: Default::default(),
        media_url: cfg.media.clone(),
        analysis: tokio::sync::Semaphore::new(2),
        label_tsvs: cfg.label_tsvs.clone(),
        analyzer: OnceCell::new(),
        vocab: OnceCell::new(),
    });
    refresh(&app).await;

    let mut api = v1::routes(cfg.role);
    if !cfg.api_only && cfg.role == Role::All {
        api = api
            .route("/api/tracks", get(tracks))
            .route("/api/recommend", post(recommend))
            .route("/api/search", post(search))
            .route(
                "/api/vocab",
                get(|| async { Json(serde_json::json!({ "genres": GENRES, "moods": MOODS })) }),
            )
            .route(
                "/api/analyze",
                post(analyze).layer(DefaultBodyLimit::max(300 << 20)),
            )
            .route("/api/audio/{id}", get(audio))
            .route("/api/stats", get(stats))
            .route("/api/eval", post(eval))
            .route("/api/labels", get(labels))
            .route("/api/ratings", get(list_ratings).post(add_rating));
    }
    api = api.route_layer(middleware::from_fn_with_state(
        Arc::new(tokio::sync::Semaphore::new(cfg.max_concurrent.max(1))),
        shed_load,
    ));
    let tokens: Vec<String> = cfg
        .api_token
        .iter()
        .flat_map(|t| t.split(','))
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    if !tokens.is_empty() {
        api = api.route_layer(middleware::from_fn_with_state(
            Arc::new(tokens),
            require_token,
        ));
    }
    let indexer = cfg.role == Role::Indexer;
    let mut router = Router::new()
        // liveness: the process answers; readiness: a snapshot is loaded (never behind the token)
        .route("/healthz", get(|| async { "ok" }))
        // Prometheus, open like the probes (restrict it with a NetworkPolicy)
        .route(
            "/metrics",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
                    crate::telemetry::prometheus().render(),
                )
            }),
        )
        .route(
            "/readyz",
            get(move |State(s): State<AppState>| async move {
                // an indexer must take pushes before any track exists
                if indexer || s.lib.load().is_some() {
                    (StatusCode::OK, "ready")
                } else {
                    (StatusCode::SERVICE_UNAVAILABLE, "no index loaded")
                }
            }),
        )
        .merge(api);
    if !cfg.api_only && cfg.role == Role::All {
        router = router.route("/", get(|| async { Html(FRONTEND) }));
    }
    // route_layer: after routing, so requests are labelled by their route template
    let router = router.route_layer(middleware::from_fn(crate::telemetry::observe));
    (router.with_state(app.clone()), app)
}

pub async fn serve(store: Store, cfg: ServeConfig) -> anyhow::Result<()> {
    // install the recorder before the first library load, so its gauges are kept
    crate::telemetry::prometheus();
    let (router, app) = router(store, &cfg).await;
    if app.lib.load().is_none() {
        tracing::warn!(
            "no snapshot in {} yet — serving 503 until `earwig index` publishes one",
            app.store.url
        );
    }
    if cfg.api_token.is_none() && cfg.host != "127.0.0.1" && cfg.host != "localhost" {
        tracing::warn!(
            "listening on {} without --api-token: anyone who can reach it can use the API",
            cfg.host
        );
    }
    let (poller, refresh_secs) = (app.clone(), cfg.refresh_secs.max(1));
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(refresh_secs)).await;
            refresh(&poller).await;
        }
    });

    tracing::info!("listening on http://{}:{}", cfg.host, cfg.port);
    let listener = tokio::net::TcpListener::bind((cfg.host.as_str(), cfg.port)).await?;
    // graceful shutdown on SIGTERM / Ctrl-C: in-flight requests finish during a rolling update
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler");
        tokio::select! { _ = ctrl_c => {}, _ = term.recv() => {} }
    }
    #[cfg(not(unix))]
    let _ = ctrl_c.await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use earwig_core::{track_id, Track};
    use tower::ServiceExt;

    /// Tracks on two "styles": a, b point one way, c, d the other, e in between.
    /// Artist X made a, b and e (b and e on album "One"); Y made c and d.
    fn tracks() -> Vec<Track> {
        let t = |key: &str, a: [f32; 2], bpm: f32| Track {
            meta: TrackMeta {
                id: track_id(key),
                path: key.into(),
                title: key.into(),
                artist: if "abe".contains(&key[..1]) { "X" } else { "Y" }.into(),
                album: if "be".contains(&key[..1]) { "One" } else { "" }.into(),
                bpm,
                ..Default::default()
            },
            audio: a.to_vec(),
            text: a.to_vec(),
            fingerprint: (1, 1),
        };
        vec![
            t("a.mp3", [1.0, 0.0], 100.0),
            t("b.mp3", [0.99, 0.14], 102.0),
            t("c.mp3", [0.0, 1.0], 140.0),
            t("d.mp3", [0.14, 0.99], 141.0),
            t("e.mp3", [0.7, 0.7], 120.0),
        ]
    }

    async fn app(token: Option<&str>, api_only: bool, publish: bool) -> Router {
        app_with(token, api_only, publish, Role::All).await
    }

    async fn app_with(token: Option<&str>, api_only: bool, publish: bool, role: Role) -> Router {
        let dir = std::env::temp_dir().join(format!(
            "earwig-api-test-{}-{}",
            std::process::id(),
            unique_suffix()
        ));
        app_in(&dir, token, api_only, publish, role).await
    }

    async fn app_in(
        dir: &std::path::Path,
        token: Option<&str>,
        api_only: bool,
        publish: bool,
        role: Role,
    ) -> Router {
        let store = Store::open(dir.to_str().unwrap()).unwrap();
        if publish {
            library::publish(
                &store,
                "t",
                &Index {
                    tracks: tracks(),
                    media: dir.display().to_string(),
                },
            )
            .await
            .unwrap();
        }
        let cfg = ServeConfig {
            host: "127.0.0.1".into(),
            port: 0,
            label_tsvs: vec![],
            refresh_secs: 30,
            api_token: token.map(String::from),
            role,
            max_concurrent: 64,
            api_only,
            media: None,
        };
        router(store, &cfg).await.0
    }

    async fn call(
        app: &Router,
        method: &str,
        uri: &str,
        body: Option<serde_json::Value>,
        token: Option<&str>,
    ) -> (StatusCode, serde_json::Value) {
        let mut req = Request::builder().method(method).uri(uri);
        if let Some(t) = token {
            req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
        }
        let req = match body {
            Some(b) => req
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(b.to_string())),
            None => req.body(Body::empty()),
        }
        .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        )
    }

    fn keys(v: &serde_json::Value) -> Vec<String> {
        v["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["key"].as_str().unwrap().to_string())
            .collect()
    }

    #[tokio::test]
    async fn pushed_tracks_override_reach_other_replicas_and_can_be_removed() {
        use serde_json::json;
        let dir = std::env::temp_dir().join(format!(
            "earwig-overlay-test-{}-{}",
            std::process::id(),
            unique_suffix()
        ));
        let app = app_in(&dir, None, true, true, Role::All).await;
        let put = |id: &str, body: serde_json::Value| {
            let (app, uri) = (app.clone(), format!("/api/v1/tracks/{id}"));
            async move { call(&app, "PUT", &uri, Some(body), None).await }
        };
        let artist = |app: Router| async move {
            call(&app, "GET", "/api/v1/tracks?key=a.mp3", None, None)
                .await
                .1["artist"]
                .clone()
        };

        // metadata only, on an indexed track (text unchanged, so no model is needed)
        let (code, v) = put("a.mp3", json!({ "artist": " Z ", "album": "Live" })).await;
        assert_eq!(code, StatusCode::OK, "{v}");
        assert_eq!(
            (v["artist"].clone(), v["album"].clone()),
            (json!("Z"), json!("Live"))
        );
        assert_eq!(artist(app.clone()).await, "Z");
        // a second replica on the same store sees it
        let (other, other_state) = replica(&dir).await;
        assert_eq!(artist(other.clone()).await, "Z");

        for (id, body) in [
            ("unknown", json!({ "artist": "Z" })),
            ("a.mp3", json!({ "media_key": "../secret" })),
            ("a.mp3", json!({ "media_key": "missing.mp3" })),
            ("a.mp3", json!({ "colour": "red" })),
        ] {
            let (code, v) = put(id, body).await;
            assert_eq!(code, StatusCode::UNPROCESSABLE_ENTITY, "{id}: {v}");
        }

        // deleting brings the scanned version back, here and on the other replica
        let del = call(&app, "DELETE", "/api/v1/tracks/a.mp3", None, None)
            .await
            .0;
        assert_eq!(del, StatusCode::NO_CONTENT);
        assert_eq!(artist(app.clone()).await, "X");
        assert_eq!(artist(other.clone()).await, "Z", "until its next refresh");
        refresh(&other_state).await;
        assert_eq!(artist(other).await, "X");
        let del = call(&app, "DELETE", "/api/v1/tracks/a.mp3", None, None)
            .await
            .0;
        assert_eq!(del, StatusCode::NOT_FOUND);
    }

    /// Another replica on `dir`, with its state to drive its refresh by hand.
    async fn replica(dir: &std::path::Path) -> (Router, Arc<App>) {
        let cfg = ServeConfig {
            host: "127.0.0.1".into(),
            port: 0,
            label_tsvs: vec![],
            refresh_secs: 30,
            api_token: None,
            role: Role::All,
            max_concurrent: 64,
            api_only: true,
            media: None,
        };
        router(Store::open(dir.to_str().unwrap()).unwrap(), &cfg).await
    }

    #[tokio::test]
    async fn a_library_of_pushed_tracks_only_is_served() {
        let dir = std::env::temp_dir().join(format!(
            "earwig-overlay-only-{}-{}",
            std::process::id(),
            unique_suffix()
        ));
        let store = Store::open(dir.to_str().unwrap()).unwrap();
        for t in tracks() {
            let one = Index {
                tracks: vec![t.clone()],
                media: String::new(),
            };
            store
                .put(&overlay_key(&t.meta.path), one.to_bytes().unwrap())
                .await
                .unwrap();
        }
        let s = replica(&dir).await.1;
        let lib = s.lib().unwrap();
        assert_eq!(lib.index.tracks.len(), 5);
        assert!(lib.media.is_none());
        assert_eq!(
            overlay_id(&overlay_key("x/é.mp3")).as_deref(),
            Some("x/é.mp3")
        );
        // removing them all empties the library: not ready any more
        for t in tracks() {
            store.delete(&overlay_key(&t.meta.path)).await.unwrap();
        }
        refresh(&s).await;
        assert!(s.lib.load().is_none());
    }

    #[tokio::test]
    async fn v1_filters_diversifies_and_weighs() {
        let app = app(None, false, true).await;
        let rec = |body: serde_json::Value| {
            let app = app.clone();
            async move {
                let (code, v) =
                    call(&app, "POST", "/api/v1/recommendations", Some(body), None).await;
                assert_eq!(code, StatusCode::OK, "{v}");
                keys(&v)
            }
        };
        use serde_json::json;
        // closest to a: b, e (both artist X, album One), then d
        assert_eq!(
            rec(json!({ "seeds": ["a.mp3"], "limit": 2 })).await,
            ["b.mp3", "e.mp3"]
        );
        assert_eq!(
            rec(json!({ "seeds": ["a.mp3"], "limit": 2, "max_per_artist": 1 })).await,
            ["b.mp3", "d.mp3"]
        );
        assert_eq!(
            rec(json!({ "seeds": ["a.mp3"], "limit": 2, "max_per_album": 1 })).await,
            ["b.mp3", "d.mp3"]
        );
        assert_eq!(
            rec(json!({ "seeds": ["a.mp3"], "candidates": ["c.mp3", "e.mp3", "nope"] })).await,
            ["e.mp3", "c.mp3"]
        );
        // e sits between b and d; moving away from a picks d
        assert_eq!(
            rec(json!({ "seeds": ["e.mp3"], "avoid": ["a.mp3"], "limit": 1 })).await,
            ["d.mp3"]
        );
        // a heavier seed pulls the profile its way
        assert_eq!(
            rec(json!({ "seeds": ["a.mp3", { "key": "c.mp3", "weight": 3 }], "limit": 1 })).await,
            ["d.mp3"]
        );
        // explained by the seed it sounds like, and the tempo (seeds' median 140, d at 141)
        let (_, v) = call(
            &app,
            "POST",
            "/api/v1/recommendations",
            Some(json!({ "seeds": ["a.mp3", { "key": "c.mp3", "weight": 3 }], "limit": 1 })),
            None,
        )
        .await;
        let e = &v["items"][0]["explain"];
        assert_eq!(e["because"]["key"], "c.mp3", "{v}");
        assert_eq!(e["driven_by"], "sound");
        assert_eq!(e["tempo"]["track_bpm"], 141.0);
        let (code, _) = call(
            &app,
            "POST",
            "/api/v1/recommendations",
            Some(json!({ "seeds": [{ "key": "a.mp3", "weight": -1 }] })),
            None,
        )
        .await;
        assert_eq!(code, StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn v1_recommends_from_seeds_with_exclusions() {
        let app = app(None, false, true).await;
        let (code, v) = call(
            &app,
            "POST",
            "/api/v1/recommendations",
            Some(serde_json::json!({ "seeds": ["a.mp3", "nope.mp3"], "limit": 2 })),
            None,
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{v}");
        assert_eq!(
            keys(&v),
            vec!["b.mp3", "e.mp3"],
            "closest first, seed excluded"
        );
        assert_eq!(v["unknown_seeds"], serde_json::json!(["nope.mp3"]));
        assert!(v["items"][0]["parts"]["sound"].as_f64().unwrap() > 0.9);
        assert_eq!(v["items"][0]["track"]["key"], "b.mp3");

        let (_, v) = call(
            &app,
            "POST",
            "/api/v1/recommendations",
            Some(serde_json::json!({ "seeds": ["a.mp3"], "exclude": ["b.mp3"], "limit": 1 })),
            None,
        )
        .await;
        assert_eq!(keys(&v), vec!["e.mp3"]);

        // two seeds from both styles: their profile is in between
        let (_, v) = call(
            &app,
            "POST",
            "/api/v1/recommendations",
            Some(serde_json::json!({ "seeds": ["a.mp3", "c.mp3"], "limit": 1 })),
            None,
        )
        .await;
        assert_eq!(keys(&v), vec!["e.mp3"]);

        // tempo only
        let (_, v) = call(
            &app,
            "POST",
            "/api/v1/recommendations",
            Some(serde_json::json!({ "bpm": 140, "limit": 2 })),
            None,
        )
        .await;
        assert_eq!(keys(&v), vec!["c.mp3", "d.mp3"]);
    }

    #[tokio::test]
    async fn v1_validates_and_looks_up_tracks() {
        let app = app(None, false, true).await;
        let (code, v) = call(
            &app,
            "POST",
            "/api/v1/recommendations",
            Some(serde_json::json!({ "seeds": ["a.mp3"], "limit": 0 })),
            None,
        )
        .await;
        assert_eq!(code, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(v["error"].as_str().unwrap().contains("limit"));
        let (code, v) = call(
            &app,
            "POST",
            "/api/v1/recommendations",
            Some(serde_json::json!({ "seeds": ["nope.mp3"] })),
            None,
        )
        .await;
        assert_eq!(
            code,
            StatusCode::UNPROCESSABLE_ENTITY,
            "only unknown seeds: {v}"
        );
        assert!(v["error"].as_str().unwrap().contains("nope.mp3"));
        let (code, _) = call(
            &app,
            "POST",
            "/api/v1/recommendations",
            Some(serde_json::json!({ "seed": ["a.mp3"] })),
            None,
        )
        .await;
        assert_eq!(
            code,
            StatusCode::UNPROCESSABLE_ENTITY,
            "unknown field rejected"
        );
        let (code, v) = call(&app, "GET", "/api/v1/tracks?key=c.mp3", None, None).await;
        assert_eq!((code, v["bpm"].as_f64()), (StatusCode::OK, Some(140.0)));
        let (code, _) = call(&app, "GET", "/api/v1/tracks?key=zzz.mp3", None, None).await;
        assert_eq!(code, StatusCode::NOT_FOUND);
        let (code, v) = call(&app, "GET", "/api/v1/openapi.json", None, None).await;
        assert_eq!(
            (code, v["openapi"].as_str()),
            (StatusCode::OK, Some("3.1.0"))
        );
    }

    #[tokio::test]
    async fn an_unreachable_media_store_does_not_block_recommendations() {
        let dir = std::env::temp_dir().join(format!(
            "earwig-no-media-{}-{}",
            std::process::id(),
            unique_suffix()
        ));
        let store = Store::open(dir.to_str().unwrap()).unwrap();
        // indexed on another machine: this folder cannot be created here
        library::publish(
            &store,
            "t",
            &Index {
                tracks: tracks(),
                media: "/earwig-no-such-root/media".into(),
            },
        )
        .await
        .unwrap();
        let app = app_in(&dir, None, true, false, Role::Api).await;
        assert_eq!(
            call(&app, "GET", "/readyz", None, None).await.0,
            StatusCode::OK
        );
        let (code, v) = call(
            &app,
            "POST",
            "/api/v1/recommendations",
            Some(serde_json::json!({ "seeds": ["a.mp3"], "limit": 1 })),
            None,
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{v}");
    }

    #[tokio::test]
    async fn roles_split_recommendations_from_analysis() {
        let rec = serde_json::json!({ "seeds": ["a.mp3"] });
        let put = serde_json::json!({ "artist": "Someone" });

        let api = app_with(None, true, true, Role::Api).await;
        let code = |r: (StatusCode, serde_json::Value)| r.0;
        assert_eq!(
            code(
                call(
                    &api,
                    "POST",
                    "/api/v1/recommendations",
                    Some(rec.clone()),
                    None
                )
                .await
            ),
            StatusCode::OK
        );
        assert_eq!(
            code(call(&api, "PUT", "/api/v1/tracks/a.mp3", Some(put.clone()), None).await),
            StatusCode::NOT_FOUND,
            "an api replica takes no pushes"
        );

        let indexer = app_with(None, true, false, Role::Indexer).await;
        assert_eq!(
            code(call(&indexer, "GET", "/readyz", None, None).await),
            StatusCode::OK,
            "an indexer is ready before any track exists, to take the first pushes"
        );
        assert_eq!(
            code(call(&indexer, "POST", "/api/v1/recommendations", Some(rec), None).await),
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn several_tokens_allow_rotation_and_metrics_are_exposed() {
        // the recorder must exist before the requests it counts
        crate::telemetry::prometheus();
        let app = app(Some("new, old"), true, true).await;
        let body = serde_json::json!({ "seeds": ["a.mp3"] });
        for t in ["new", "old"] {
            let (code, _) = call(
                &app,
                "POST",
                "/api/v1/recommendations",
                Some(body.clone()),
                Some(t),
            )
            .await;
            assert_eq!(code, StatusCode::OK, "token {t}");
        }
        let (code, _) = call(
            &app,
            "POST",
            "/api/v1/recommendations",
            Some(body),
            Some("other"),
        )
        .await;
        assert_eq!(code, StatusCode::UNAUTHORIZED);

        // /metrics is open like the probes, and counts requests by route template
        let res = app
            .clone()
            .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let text =
            String::from_utf8(to_bytes(res.into_body(), 1 << 20).await.unwrap().to_vec()).unwrap();
        assert!(
            text.contains("earwig_http_requests_total")
                && text.contains("route=\"/api/v1/recommendations\""),
            "{text}"
        );
    }

    #[tokio::test]
    async fn token_guards_the_api_but_not_the_probes() {
        let app = app(Some("s3cret"), false, true).await;
        let body = serde_json::json!({ "seeds": ["a.mp3"] });
        assert_eq!(
            call(
                &app,
                "POST",
                "/api/v1/recommendations",
                Some(body.clone()),
                None
            )
            .await
            .0,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            call(
                &app,
                "POST",
                "/api/v1/recommendations",
                Some(body.clone()),
                Some("wrong")
            )
            .await
            .0,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            call(
                &app,
                "POST",
                "/api/v1/recommendations",
                Some(body),
                Some("s3cret")
            )
            .await
            .0,
            StatusCode::OK
        );
        assert_eq!(
            call(&app, "GET", "/api/stats", None, None).await.0,
            StatusCode::UNAUTHORIZED,
            "legacy routes too"
        );
        assert_eq!(
            call(&app, "GET", "/healthz", None, None).await.0,
            StatusCode::OK
        );
        assert_eq!(
            call(&app, "GET", "/readyz", None, None).await.0,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn api_only_hides_the_ui_and_legacy_routes() {
        let app = app(None, true, true).await;
        assert_eq!(
            call(&app, "GET", "/", None, None).await.0,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            call(&app, "GET", "/api/stats", None, None).await.0,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            call(&app, "GET", "/api/v1/tracks?key=a.mp3", None, None)
                .await
                .0,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn without_a_snapshot_it_is_alive_but_not_ready() {
        let app = app(None, false, false).await;
        assert_eq!(
            call(&app, "GET", "/healthz", None, None).await.0,
            StatusCode::OK
        );
        assert_eq!(
            call(&app, "GET", "/readyz", None, None).await.0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        let (code, v) = call(
            &app,
            "POST",
            "/api/v1/recommendations",
            Some(serde_json::json!({ "bpm": 120 })),
            None,
        )
        .await;
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert!(v["error"].is_string());
    }
}
