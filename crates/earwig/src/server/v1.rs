//! `/api/v1` — the stable API for other services. Tracks are named by their **media key**
//! (their path in the media bucket / folder, e.g. `tracks/42.mp3`), so a caller never needs
//! earwig's internal ids. Contract: [`OPENAPI`], served at `/api/v1/openapi.json`.
//!
//! * `POST /api/v1/recommendations` — from seeds (keys, e.g. a user's history), uploaded
//!   features, moods, styles, a description and/or a tempo; with exclusions.
//! * `GET  /api/v1/tracks?key=…`     — what earwig knows about one track.
//! * `POST /api/v1/analyze?name=…`   — features of a file that is not in the library.
//! * `PUT  /api/v1/tracks/{id}`      — add or update a track under the app's own id (its audio,
//!   a media key, or new metadata only); recommendable right away. `DELETE` removes it.
//!
//! Errors are `{"error": "…"}` with a 4xx / 5xx status.

use std::collections::HashSet;

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Path, Query as UrlQuery, State},
    http::{header, HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use earwig_core::{
    scoring::{bpm_delta, cosine, Scorable},
    track_id, Index, Query, Track, TrackMeta,
};
use serde::{Deserialize, Serialize};

use super::{
    analyze_bytes, analyze_upload, centroid, default_tolerance, embed_text, overlay_key, refresh,
    run_search, ApiError, AppState, Role, SearchSpec,
};
use crate::storage::Store;

pub const OPENAPI: &str = include_str!("openapi.json");

const MAX_SEEDS: usize = 1000;
const MAX_EXCLUDE: usize = 100_000;
const MAX_FEATURES: usize = 50;
const MAX_LABELS: usize = 20;
const MAX_TEXT: usize = 500;
const MAX_CANDIDATES: usize = 100_000;

/// API errors as JSON.
struct V1Error(StatusCode, String);

impl From<ApiError> for V1Error {
    fn from((code, msg): ApiError) -> Self {
        Self(code, msg)
    }
}

impl IntoResponse for V1Error {
    fn into_response(self) -> axum::response::Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

fn invalid(msg: impl Into<String>) -> V1Error {
    V1Error(StatusCode::UNPROCESSABLE_ENTITY, msg.into())
}

/// The public view of a track.
#[derive(Serialize, Debug)]
pub struct V1Track {
    key: String,
    title: String,
    artist: String,
    album: String,
    genre: String,
    subgenre: String,
    year: Option<i64>,
    bpm: f32,
    musical_key: String,
    duration_s: f32,
    auto_genres: Vec<String>,
    auto_moods: Vec<String>,
}

impl From<&TrackMeta> for V1Track {
    fn from(m: &TrackMeta) -> Self {
        Self {
            key: m.path.clone(),
            title: m.title.clone(),
            artist: m.artist.clone(),
            album: m.album.clone(),
            genre: m.genre.clone(),
            subgenre: m.subgenre.clone(),
            year: m.year,
            bpm: m.bpm,
            musical_key: m.key.clone(),
            duration_s: m.duration_s,
            auto_genres: m.auto_genres.clone(),
            auto_moods: m.auto_moods.clone(),
        }
    }
}

/// Weights of the parts; an absent weight defaults to a sensible value when its input is given.
#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
struct V1Weights {
    sound: Option<f32>,
    tags: Option<f32>,
    mood: Option<f32>,
    genre: Option<f32>,
    description: Option<f32>,
    bpm: Option<f32>,
    avoid: Option<f32>,
}

/// A seed: a media key, or `{key, weight}` (weight 1 when left out; e.g. more for a liked track).
#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum Seed {
    Key(String),
    Weighted(WeightedSeed),
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct WeightedSeed {
    key: String,
    #[serde(default = "one")]
    weight: f32,
}
fn one() -> f32 {
    1.0
}

impl Seed {
    fn key_weight(&self) -> (&str, f32) {
        match self {
            Seed::Key(k) => (k, 1.0),
            Seed::Weighted(s) => (&s.key, s.weight),
        }
    }
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct RecommendationsRequest {
    /// media keys of reference tracks (e.g. a user's recent listens); their profile is averaged
    #[serde(default)]
    seeds: Vec<Seed>,
    /// media keys to move away from (skips, dislikes); never returned
    #[serde(default)]
    avoid: Vec<String>,
    /// features returned by /api/v1/analyze, used like seeds
    #[serde(default)]
    features: Vec<Query>,
    #[serde(default)]
    moods: Vec<String>,
    #[serde(default)]
    genres: Vec<String>,
    #[serde(default)]
    description: String,
    bpm: Option<f32>,
    #[serde(default)]
    weights: V1Weights,
    #[serde(default = "default_tolerance")]
    bpm_tolerance: f32,
    /// media keys never returned (seeds are always excluded)
    #[serde(default)]
    exclude: Vec<String>,
    /// when given, only these media keys may be returned (the app's visibility rules)
    candidates: Option<Vec<String>>,
    /// at most this many results per artist / per album
    max_per_artist: Option<usize>,
    max_per_album: Option<usize>,
    #[serde(default = "default_limit")]
    limit: usize,
}
fn default_limit() -> usize {
    20
}

#[derive(Serialize)]
struct Item {
    key: String,
    score: f32,
    parts: serde_json::Map<String, serde_json::Value>,
    explain: Explain,
    track: V1Track,
}

/// Why a track was recommended, for display ("Because you listened to X · same tempo").
#[derive(Serialize, Debug, PartialEq)]
struct Explain {
    /// the part that added most to the score: sound, tags, mood, genre, description, bpm, avoid
    driven_by: String,
    /// the seed this track sounds most like (when there are seeds)
    #[serde(skip_serializing_if = "Option::is_none")]
    because: Option<Because>,
    /// tempos, when they match (within the tolerance, half / double time included)
    #[serde(skip_serializing_if = "Option::is_none")]
    tempo: Option<Tempo>,
    /// labels the track shares with the seeds or the requested moods / genres, from human
    /// metadata only (file tags or the app's): CLAP auto-tags are too unreliable to show
    shared_labels: Vec<String>,
}

#[derive(Serialize, Debug, PartialEq)]
struct Because {
    key: String,
    title: String,
    artist: String,
    /// audio similarity to that seed (cosine)
    similarity: f32,
}

#[derive(Serialize, Debug, PartialEq)]
struct Tempo {
    query_bpm: f32,
    track_bpm: f32,
}

const MAX_SHARED: usize = 5;

/// Human labels of a track, lowercased.
fn human_labels(m: &TrackMeta) -> Vec<String> {
    [&m.genre, &m.subgenre]
        .into_iter()
        .chain(&m.moods)
        .chain(&m.themes)
        .chain(&m.tags)
        .map(|l| l.trim().to_lowercase())
        .filter(|l| !l.is_empty())
        .collect()
}

fn explain(
    track: &Track,
    driven_by: &str,
    seeds: &[&Track],
    wanted: &HashSet<String>,
    query_bpm: Option<f32>,
    bpm_tolerance: f32,
) -> Explain {
    let because = seeds
        .iter()
        .map(|s| (s, cosine(&s.audio, &track.audio)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(s, similarity)| Because {
            key: s.meta.path.clone(),
            title: s.meta.title.clone(),
            artist: s.meta.artist.clone(),
            similarity,
        });
    let tempo = query_bpm
        .filter(|&q| q > 0.0 && track.meta.bpm > 0.0)
        .filter(|&q| bpm_delta(q, track.meta.bpm) <= bpm_tolerance)
        .map(|q| Tempo {
            query_bpm: q,
            track_bpm: track.meta.bpm,
        });
    let mut shared_labels = Vec::new();
    for l in human_labels(&track.meta) {
        if wanted.contains(&l) && !shared_labels.contains(&l) && shared_labels.len() < MAX_SHARED {
            shared_labels.push(l);
        }
    }
    Explain {
        driven_by: if driven_by == "track" {
            "sound"
        } else {
            driven_by
        }
        .to_string(),
        because,
        tempo,
        shared_labels,
    }
}

#[derive(Serialize)]
struct RecommendationsResponse {
    items: Vec<Item>,
    /// seeds and avoided keys that are not in the current snapshot (not indexed yet, or removed)
    unknown_seeds: Vec<String>,
    snapshot: String,
    took_ms: f64,
}

fn validate(r: &RecommendationsRequest) -> Result<(), V1Error> {
    let labels_ok = |v: &[String]| v.len() <= MAX_LABELS && v.iter().all(|l| l.len() <= 64);
    if r.seeds.len() > MAX_SEEDS || r.avoid.len() > MAX_SEEDS {
        return Err(invalid(format!(
            "at most {MAX_SEEDS} seeds and {MAX_SEEDS} avoided keys"
        )));
    }
    if r.seeds
        .iter()
        .any(|s| !s.key_weight().1.is_finite() || s.key_weight().1 < 0.0)
    {
        return Err(invalid(
            "seed weights must be finite and ≥ 0 (use `avoid` to move away)",
        ));
    }
    if r.candidates
        .as_ref()
        .is_some_and(|c| c.len() > MAX_CANDIDATES)
    {
        return Err(invalid(format!("at most {MAX_CANDIDATES} candidates")));
    }
    if r.max_per_artist == Some(0) || r.max_per_album == Some(0) {
        return Err(invalid("max_per_artist / max_per_album: at least 1"));
    }
    if r.exclude.len() > MAX_EXCLUDE {
        return Err(invalid(format!("at most {MAX_EXCLUDE} excluded keys")));
    }
    if r.features.len() > MAX_FEATURES {
        return Err(invalid(format!("at most {MAX_FEATURES} feature sets")));
    }
    if !labels_ok(&r.moods) || !labels_ok(&r.genres) {
        return Err(invalid(format!(
            "at most {MAX_LABELS} moods / genres of at most 64 characters"
        )));
    }
    if r.description.chars().count() > MAX_TEXT {
        return Err(invalid(format!(
            "description: at most {MAX_TEXT} characters"
        )));
    }
    if !(1..=100).contains(&r.limit) {
        return Err(invalid("limit: 1 to 100"));
    }
    let weights = [
        r.weights.sound,
        r.weights.tags,
        r.weights.mood,
        r.weights.genre,
        r.weights.description,
        r.weights.bpm,
        r.weights.avoid,
    ];
    if weights.iter().flatten().any(|w| !w.is_finite() || *w < 0.0)
        || !r.bpm_tolerance.is_finite()
        || r.bpm.is_some_and(|b| !b.is_finite())
    {
        return Err(invalid(
            "weights, bpm and bpm_tolerance must be finite, weights ≥ 0",
        ));
    }
    Ok(())
}

async fn recommendations(
    State(s): State<AppState>,
    Json(req): Json<RecommendationsRequest>,
) -> Result<Json<RecommendationsResponse>, V1Error> {
    validate(&req)?;
    let lib = s.lib()?;
    let tracks = &lib.index.tracks;

    let mut unknown_seeds = Vec::new();
    let mut exclude = HashSet::new();
    // `refs` must not live across the await below (trait objects are not Sync)
    // seeds that count, for the explanations
    let mut seed_ix: Vec<usize> = Vec::new();
    let (reference, avoid) = {
        let mut refs: Vec<(&dyn Scorable, f32)> = Vec::new();
        for seed in &req.seeds {
            let (key, weight) = seed.key_weight();
            match lib.by_key.get(key) {
                Some(&i) => {
                    refs.push((&tracks[i], weight));
                    exclude.insert(i);
                    if weight > 0.0 {
                        seed_ix.push(i);
                    }
                }
                None => unknown_seeds.push(key.to_string()),
            }
        }
        refs.extend(req.features.iter().map(|q| (q as &dyn Scorable, 1.0)));
        let mut avoided: Vec<(&dyn Scorable, f32)> = Vec::new();
        for key in &req.avoid {
            match lib.by_key.get(key) {
                Some(&i) => {
                    avoided.push((&tracks[i], 1.0));
                    exclude.insert(i);
                }
                None => unknown_seeds.push(key.clone()),
            }
        }
        (centroid(&refs), centroid(&avoided))
    };
    let allowed = req.candidates.as_ref().map(|keys| {
        keys.iter()
            .filter_map(|k| lib.by_key.get(k).copied())
            .collect::<HashSet<usize>>()
    });
    exclude.extend(
        req.exclude
            .iter()
            .filter_map(|k| lib.by_key.get(k).copied()),
    );

    let w = &req.weights;
    let has_ref = reference.is_some();
    let default = |given: Option<f32>, used: bool, value: f32| {
        given.unwrap_or(if used { value } else { 0.0 })
    };
    let weights = [
        default(w.sound, has_ref, 0.7),
        default(w.tags, has_ref, 0.3),
        default(w.mood, !req.moods.is_empty(), 1.0),
        default(w.genre, !req.genres.is_empty(), 1.0),
        default(w.description, !req.description.trim().is_empty(), 1.0),
        default(w.bpm, req.bpm.is_some(), 0.5),
        default(w.avoid, avoid.is_some(), 0.5),
    ];
    // for the explanations, before the request moves into the search
    let req_labels: Vec<String> = req.moods.iter().chain(&req.genres).cloned().collect();
    let query_bpm = req.bpm.or(reference.as_ref().map(|r| r.bpm));
    let bpm_tolerance = req.bpm_tolerance;
    let spec = SearchSpec {
        reference,
        exclude,
        allowed,
        avoid,
        max_per_artist: req.max_per_artist,
        max_per_album: req.max_per_album,
        moods: req.moods,
        genres: req.genres,
        description: req.description,
        bpm: req.bpm,
        weights,
        bpm_tolerance: req.bpm_tolerance,
        k: req.limit,
    };
    let (hits, took_ms) = run_search(&s, &lib, spec).await.map_err(|(code, msg)| {
        if code == StatusCode::BAD_REQUEST && !unknown_seeds.is_empty() {
            invalid(format!(
                "{msg} (unknown seeds: {})",
                unknown_seeds.join(", ")
            ))
        } else {
            V1Error(code, msg)
        }
    })?;
    let seeds: Vec<&Track> = seed_ix.iter().map(|&i| &tracks[i]).collect();
    let wanted: HashSet<String> = seeds
        .iter()
        .flat_map(|s| human_labels(&s.meta))
        .chain(req_labels.iter().map(|l| l.trim().to_lowercase()))
        .collect();
    let items = hits
        .into_iter()
        .map(|h| {
            let meta = &tracks[h.index].meta;
            // public part names: "sound" is the reference's audio similarity
            let parts = h
                .parts
                .iter()
                .map(|(n, v)| {
                    (
                        if *n == "track" { "sound" } else { n }.to_string(),
                        (*v).into(),
                    )
                })
                .collect();
            Item {
                key: meta.path.clone(),
                score: h.score,
                parts,
                explain: explain(
                    &tracks[h.index],
                    h.driven_by,
                    &seeds,
                    &wanted,
                    query_bpm,
                    bpm_tolerance,
                ),
                track: meta.into(),
            }
        })
        .collect();
    Ok(Json(RecommendationsResponse {
        items,
        unknown_seeds,
        snapshot: lib.key.clone(),
        took_ms,
    }))
}

#[derive(Deserialize)]
struct KeyParam {
    key: String,
}

async fn track(
    State(s): State<AppState>,
    UrlQuery(p): UrlQuery<KeyParam>,
) -> Result<Json<V1Track>, V1Error> {
    let lib = s.lib()?;
    let &i = lib.by_key.get(&p.key).ok_or_else(|| {
        V1Error(
            StatusCode::NOT_FOUND,
            format!("no track with key {}", p.key),
        )
    })?;
    Ok(Json((&lib.index.tracks[i].meta).into()))
}

#[derive(Deserialize)]
struct NameParam {
    name: Option<String>,
}

async fn analyze(
    State(s): State<AppState>,
    UrlQuery(p): UrlQuery<NameParam>,
    body: Bytes,
) -> Result<Json<serde_json::Value>, V1Error> {
    if body.is_empty() {
        return Err(invalid(
            "empty body: send the audio file as the request body",
        ));
    }
    let (meta, features) = analyze_upload(&s, p.name, body).await?;
    Ok(Json(
        serde_json::json!({ "track": V1Track::from(&meta), "features": features }),
    ))
}

/// A pushed track's metadata, overriding the file's tags. As a JSON body, `media_key` names the
/// audio in the media store; as query parameters (with the audio as the body), lists are absent.
#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
struct TrackFields {
    media_key: Option<String>,
    title: Option<String>,
    artist: Option<String>,
    album: Option<String>,
    genre: Option<String>,
    subgenre: Option<String>,
    year: Option<i64>,
    moods: Option<Vec<String>>,
    themes: Option<Vec<String>>,
    tags: Option<Vec<String>>,
}

fn validate_track(id: &str, f: &TrackFields) -> Result<(), V1Error> {
    if id.is_empty() || id.len() > 512 || id.chars().any(char::is_control) {
        return Err(invalid("id: 1 to 512 characters, no control characters"));
    }
    let texts = [&f.title, &f.artist, &f.album, &f.genre, &f.subgenre];
    if texts.iter().flat_map(|t| t.iter()).any(|t| t.len() > 512) {
        return Err(invalid(
            "title / artist / album / genre / subgenre: at most 512 bytes",
        ));
    }
    let lists = [&f.moods, &f.themes, &f.tags];
    if lists
        .iter()
        .flat_map(|l| l.iter())
        .any(|l| l.len() > 50 || l.iter().any(|x| x.len() > 64))
    {
        return Err(invalid(
            "moods / themes / tags: at most 50 of at most 64 bytes",
        ));
    }
    if let Some(k) = &f.media_key {
        // the key is read from the media store: no escaping it
        if k.is_empty() || k.len() > 1024 || k.starts_with('/') || k.split('/').any(|p| p == "..") {
            return Err(invalid(
                "media_key: a relative key without `..`, at most 1024 bytes",
            ));
        }
    }
    Ok(())
}

/// `PUT /api/v1/tracks/{id}`: JSON metadata (+ optional `media_key`), or the audio file as the
/// body with metadata in the query string. Without audio nor key, updates an indexed track's
/// metadata. The track is stored in the overlay and served at once by this replica, by the
/// others after their next refresh.
async fn put_track(
    State(s): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    UrlQuery(query): UrlQuery<TrackFields>,
    body: Bytes,
) -> Result<Json<V1Track>, V1Error> {
    let json = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("application/json"));
    let (fields, audio) = if json {
        let f: TrackFields =
            serde_json::from_slice(&body).map_err(|e| invalid(format!("body: {e}")))?;
        (f, None)
    } else {
        if query.media_key.is_some() && !body.is_empty() {
            return Err(invalid(
                "send either the audio as the body or a media_key, not both",
            ));
        }
        (query, (!body.is_empty()).then_some(body))
    };
    validate_track(&id, &fields)?;
    let lib = s.lib.load_full();

    let existing = lib
        .as_ref()
        .and_then(|l| l.by_key.get(&id).map(|&i| l.index.tracks[i].clone()));
    let (mut meta, sound, fingerprint, analysed) = if let Some(bytes) = audio {
        let size = bytes.len() as u64;
        let (meta, sound) = analyze_bytes(&s, &id, bytes).await?;
        (meta, sound, (size, 0), true)
    } else if let Some(key) = &fields.media_key {
        let url = lib
            .as_ref()
            .map(|l| l.index.media.clone())
            .filter(|m| !m.is_empty())
            .or_else(|| s.media_url.clone())
            .ok_or_else(|| invalid("no media store: start the server with --media"))?;
        let bytes = Store::open(&url)
            .map_err(|e| V1Error(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?
            .get(key)
            .await
            .map_err(|e| invalid(format!("media_key {key}: {e:#}")))?;
        let size = bytes.len() as u64;
        let (meta, sound) = analyze_bytes(&s, key, bytes.into()).await?;
        (meta, sound, (size, 0), true)
    } else if let Some(t) = &existing {
        (t.meta.clone(), t.audio.clone(), t.fingerprint, false)
    } else {
        return Err(invalid(format!(
            "no track with id {id}: send its audio as the body, or a media_key"
        )));
    };
    if let Some(dim) = lib.as_ref().map(|l| l.index.tracks[0].audio.len()) {
        if sound.len() != dim {
            return Err(invalid("the analysis models differ from the library's"));
        }
    }

    let before = meta.text_document();
    let set = |field: &mut String, v: &Option<String>| {
        if let Some(v) = v {
            *field = v.trim().to_string();
        }
    };
    set(&mut meta.title, &fields.title);
    set(&mut meta.artist, &fields.artist);
    set(&mut meta.album, &fields.album);
    set(&mut meta.genre, &fields.genre);
    set(&mut meta.subgenre, &fields.subgenre);
    if fields.year.is_some() {
        meta.year = fields.year;
    }
    for (field, v) in [
        (&mut meta.moods, &fields.moods),
        (&mut meta.themes, &fields.themes),
        (&mut meta.tags, &fields.tags),
    ] {
        if let Some(v) = v {
            *field = v.clone();
        }
    }
    meta.path = id.clone();
    meta.id = track_id(&id);
    // the text vector only depends on the text document: re-embed when it changed
    let text = match &existing {
        Some(t) if !analysed && meta.text_document() == before => t.text.clone(),
        _ => embed_text(&s, &meta).await?,
    };

    let out = V1Track::from(&meta);
    let index = Index {
        tracks: vec![Track {
            meta,
            audio: sound,
            text,
            fingerprint,
        }],
        media: String::new(),
    };
    let bytes = index
        .to_bytes()
        .map_err(|e| V1Error(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
    s.store
        .put(&overlay_key(&id), bytes)
        .await
        .map_err(|e| V1Error(StatusCode::SERVICE_UNAVAILABLE, format!("{e:#}")))?;
    refresh(&s).await;
    Ok(Json(out))
}

/// `DELETE /api/v1/tracks/{id}`: removes a pushed track (a scanned file with the same key comes
/// back from the snapshot).
async fn delete_track(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, V1Error> {
    let key = overlay_key(&id);
    let found = s
        .store
        .get_opt(&key)
        .await
        .map_err(|e| V1Error(StatusCode::SERVICE_UNAVAILABLE, format!("{e:#}")))?;
    if found.is_none() {
        return Err(V1Error(
            StatusCode::NOT_FOUND,
            format!("no pushed track with id {id}"),
        ));
    }
    s.store
        .delete(&key)
        .await
        .map_err(|e| V1Error(StatusCode::SERVICE_UNAVAILABLE, format!("{e:#}")))?;
    refresh(&s).await;
    Ok(StatusCode::NO_CONTENT)
}

pub(super) fn routes(role: Role) -> axum::Router<AppState> {
    let mut r = Router::new().route("/api/v1/tracks", get(track)).route(
        "/api/v1/openapi.json",
        get(|| async { ([(header::CONTENT_TYPE, "application/json")], OPENAPI) }),
    );
    if role != Role::Indexer {
        r = r.route("/api/v1/recommendations", post(recommendations));
    }
    if role != Role::Api {
        r = r
            .route(
                "/api/v1/tracks/{id}",
                axum::routing::put(put_track)
                    .delete(delete_track)
                    .layer(DefaultBodyLimit::max(300 << 20)),
            )
            .route(
                "/api/v1/analyze",
                post(analyze).layer(DefaultBodyLimit::max(300 << 20)),
            );
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(key: &str, audio: [f32; 2], bpm: f32, genre: &str, moods: &[&str]) -> Track {
        Track {
            meta: TrackMeta {
                path: key.into(),
                title: key.into(),
                artist: "A".into(),
                genre: genre.into(),
                moods: moods.iter().map(|m| m.to_string()).collect(),
                auto_genres: vec!["bossa nova".into()],
                bpm,
                ..Default::default()
            },
            audio: audio.to_vec(),
            text: audio.to_vec(),
            fingerprint: (0, 0),
        }
    }

    #[test]
    fn explains_with_closest_seed_tempo_and_human_labels_only() {
        let rap = track("rap.mp3", [1.0, 0.0], 90.0, "Hip Hop", &["dark"]);
        let jazz = track("jazz.mp3", [0.0, 1.0], 120.0, "Jazz", &[]);
        let hit = track("hit.mp3", [0.9, 0.1], 181.0, "hip hop", &["Dark", "chill"]);
        let wanted: HashSet<String> = [&rap, &jazz]
            .iter()
            .flat_map(|s| human_labels(&s.meta))
            .chain(["chill".to_string()])
            .collect();

        let e = explain(&hit, "track", &[&rap, &jazz], &wanted, Some(90.0), 20.0);

        assert_eq!(e.driven_by, "sound");
        assert_eq!(e.because.as_ref().unwrap().key, "rap.mp3");
        // 181 bpm is double time of 90: a match
        assert_eq!(
            e.tempo,
            Some(Tempo {
                query_bpm: 90.0,
                track_bpm: 181.0
            })
        );
        // case-insensitive, auto-tags ("bossa nova") never shown
        assert_eq!(e.shared_labels, ["hip hop", "dark", "chill"]);

        let far = explain(&hit, "mood", &[], &HashSet::new(), Some(140.0), 20.0);
        assert!(far.because.is_none() && far.tempo.is_none() && far.shared_labels.is_empty());
        assert_eq!(far.driven_by, "mood");
    }
}
