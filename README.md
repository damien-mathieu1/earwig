<div align="center">
<h1>earwig</h1>
Content-based music recommendation: it listens to your tracks and finds the ones that sound alike 🦀

<div>

  ![Stars](https://img.shields.io/github/stars/damien-mathieu1/earwig?logo=github) ![Downloads](https://img.shields.io/crates/d/earwig?logo=rust) [![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-yellow.svg)](#license) [![GitHub release](https://img.shields.io/github/v/release/damien-mathieu1/earwig?color=success)](https://github.com/damien-mathieu1/earwig/releases/latest) [![CI](https://github.com/damien-mathieu1/earwig/actions/workflows/ci.yml/badge.svg)](https://github.com/damien-mathieu1/earwig/actions/workflows/ci.yml)
  </div>
</div>

Point earwig at a music folder or bucket: it analyses every track (timbre, style, mood, tempo) and
recommends by **sound**, with no listening history, no Python and no cloud. Use it from the CLI,
a built-in web UI, as a Rust library, or as a stateless HTTP service next to your app's own
recommendation service.

## Installation

```bash
brew install damien-mathieu1/tap/earwig
```

Or with Cargo (needs `cmake`, which builds the bundled libopus; `--no-default-features` drops Opus
and cmake):

```bash
cargo install earwig
```

Or grab a prebuilt binary (Linux x86_64 / arm64 with glibc ≥ 2.38 — Ubuntu 24.04+, Debian 13+ —
and macOS Apple Silicon) from the [releases page](https://github.com/damien-mathieu1/earwig/releases/latest),
or the container image, models included:

```bash
docker run --rm -v ~/Music:/music:ro -v earwig:/data ghcr.io/damien-mathieu1/earwig index /music
```

```bash
docker run --rm -p 8081:8081 -v earwig:/data ghcr.io/damien-mathieu1/earwig   # serve on :8081
```

The analysis models (~700 MB) are downloaded once, on the first `index` or with `earwig models`,
into `~/.cache/earwig` (`EARWIG_CACHE`).

## Quick Start

```bash
earwig index ~/Music                 # analyse the folder (later runs only touch new / changed files)
```

```bash
earwig similar ~/Downloads/song.mp3  # the 10 tracks of your library closest to any audio file
```

```bash
earwig serve                         # web UI and API on http://127.0.0.1:8081
```

```bash
earwig eval --by artist              # how often the top-10 shares the query's artist, vs random
```

It reads MP3, FLAC, WAV, Ogg/Vorbis, Opus (Ogg or WebM), M4A/AAC/ALAC, AIFF… Files without an
extension (apps often store uploads under an id) are recognised from their content.

## What it does

| | |
|---|---|
| **Sounds like** | tracks closest to one or several seeds (a user's history), weighted, with tracks to move away from |
| **Moods and styles** | "sad", "energetic", "hip hop", "reggae"… or a free English description, matched against the audio itself |
| **Tempo** | target BPM, half / double time aware |
| **Mixes** | any combination of the above, each part standardised so they mix fairly |
| **Your rules** | exclusions, allowed candidates (visibility, rights), at most N per artist / album |
| **Explanations** | for each result: the seed it sounds like, matching tempo, shared labels, the part that led |
| **Evaluation** | `earwig eval`, a benchmark page and a blind listening test |

## Using earwig from your services

Run earwig as an internal service; your recommendation service calls it over HTTP. `/api/v1` is the
stable contract — OpenAPI 3.1 at [`/api/v1/openapi.json`](crates/earwig/src/server/openapi.json),
generate a client from it. Tracks are named by **your ids** (or their path in the media bucket).

**1. Push tracks as your app receives them** — recommendable at once, your metadata overrides the
file's tags:

```bash
curl -X PUT http://earwig-indexer/api/v1/tracks/3f2a9c1e -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"media_key": "tracks/3f2a9c1e/256k.opus", "title": "Astrid", "artist": "JRK 19", "album": "…", "genre": "rap"}'
```

Or send the audio itself as the body (`?title=…&artist=…`), or index a whole folder / bucket with
`earwig index` (incremental, shardable). `DELETE /api/v1/tracks/{id}` removes a pushed track.

**2. Ask for recommendations:**

```bash
curl http://earwig/api/v1/recommendations -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' -d '{
  "seeds":   ["3f2a9c1e", "98b1", {"key": "77ac", "weight": 2}],
  "avoid":   ["44de"],
  "exclude": ["7f00"],
  "moods":   ["energetic"],
  "max_per_album": 1,
  "limit":   20
}'
```

```json
{ "items": [ { "key": "55aa", "score": 2.1,
               "parts": { "sound": 0.91, "tags": 0.88, "mood": 0.64 },
               "explain": { "driven_by": "sound",
                            "because": { "key": "77ac", "title": "Astrid", "artist": "JRK 19", "similarity": 0.93 },
                            "tempo": { "query_bpm": 122.0, "track_bpm": 124.0 },
                            "shared_labels": ["rap"] },
               "track": { "key": "55aa", "title": "…", "artist": "…", "album": "…", "bpm": 124.0, … } } ],
  "unknown_seeds": [], "snapshot": "snapshots/r001794992027.bin", "took_ms": 1.3 }
```

| Route | |
|---|---|
| `POST /api/v1/recommendations` | from `seeds` (keys or `{key, weight}`, averaged), `features` (from `analyze`), `moods`, `genres`, a `description`, a `bpm`; `avoid` (skips, dislikes), `exclude`, `candidates` (only these may be returned), `max_per_artist` / `max_per_album`; optional `weights` `{sound, tags, mood, genre, description, bpm, avoid}`; `limit` 1–100, filled after filters and caps. Seeds not indexed yet come back in `unknown_seeds` |
| `GET /api/v1/tracks?key=…` | what earwig knows about one track (404 if not indexed) |
| `PUT /api/v1/tracks/{id}` | add or update a track: audio body, JSON `{media_key, title, artist, album, genre, subgenre, year, moods, themes, tags}`, or metadata only for a known track |
| `DELETE /api/v1/tracks/{id}` | remove a pushed track |
| `POST /api/v1/analyze?name=song.mp3` | audio body → `{track, features}` for a file outside the library |

`explain` only shows labels from human metadata (yours or the files'): the zero-shot auto-tags are
useful for search but too unreliable to show to a listener. Errors are `{"error": "…"}` with a
4xx / 5xx status.

## Running in production

`earwig serve` keeps no state: the library lives in an S3-compatible bucket (AWS, MinIO, RustFS,
R2, Ceph…) or a directory, as immutable snapshots plus one object per pushed track. Every replica
loads it into RAM, polls for changes (`--refresh`) and swaps them in without downtime.

```
             ┌──────────────── S3 bucket ────────────────┐
             │ media/…          your audio files         │
             │ library/latest → snapshots/<run>.bin      │
             │ library/overlay/<id>   pushed tracks      │
             └───▲──────────────────▲───────────────▲────┘
  earwig index (CronJob,     earwig-indexer          earwig (api)
  optional, sharded)         PUT / DELETE, analyses  recommendations, N replicas
```

| | |
|---|---|
| **Roles** | `--role api` serves recommendations and never loads the analysis models; `--role indexer` takes pushes and analyses audio (at most 2 files at a time per replica). Run them apart so a burst of uploads never slows recommendations. `--role all` (default) does both |
| **Probes** | `/healthz` (alive), `/readyz` (a library is loaded; an indexer is always ready, to take the first push) |
| **Metrics** | Prometheus at `/metrics`: `earwig_http_requests_total` and `earwig_http_request_duration_seconds` per route and status, `earwig_search_duration_seconds`, `earwig_analysis_duration_seconds`, `earwig_analyses_total{result}`, `earwig_tracks`, `earwig_tracks_pushed`, `earwig_library_load_seconds`, `earwig_refresh_errors_total`, `earwig_http_shed_total` |
| **Logs** | `--log-format json` (`EARWIG_LOG_FORMAT`): one JSON object per line, one line per API request; level from `RUST_LOG` |
| **Auth** | `--api-token` (`EARWIG_API_TOKEN`): `Authorization: Bearer` on every `/api` route, compared in constant time. Several tokens comma-separated (`new,old`) rotate without downtime. Probes and metrics stay open |
| **Overload** | past `--max-concurrent` requests in flight (256), `/api` answers 503 with `Retry-After: 1` instead of queueing |
| **Shutdown** | SIGTERM finishes in-flight requests |
| **API only** | `--api-only`: no web UI, benchmark, ratings or legacy routes |

**Kubernetes** ([`deploy/kubernetes`](deploy/kubernetes)): an `earwig` Deployment (role api, 2
replicas, rolling updates with `maxUnavailable: 0`, PodDisruptionBudget, spread over nodes), an
`earwig-indexer` Deployment, an optional indexing CronJob (Indexed Job of N shards), Prometheus
scrape annotations, JSON logs, read-only root filesystem, non-root user, and a NetworkPolicy
admitting only pods labelled `earwig.io/client: "true"` (and Prometheus from `monitoring`).

```bash
kubectl -n earwig create secret generic earwig-s3 --from-literal=AWS_ACCESS_KEY_ID=… --from-literal=AWS_SECRET_ACCESS_KEY=…
kubectl -n earwig create secret generic earwig-api --from-literal=token=$(openssl rand -hex 32)
kubectl apply -k deploy/kubernetes/base     # after pointing base/config.yaml at your bucket
```

Your service then calls `http://earwig.earwig.svc` (recommendations) and
`http://earwig-indexer.earwig.svc` (pushes). `deploy/kubernetes/dev` runs everything on kind or
minikube with an in-cluster S3.

**Docker Compose** ([`docker-compose.yml`](docker-compose.yml)): S3 (RustFS), 2 index shards and 2
replicas behind nginx on http://127.0.0.1:8080 — `cargo run --release --bin seed` for demo clips, then
`docker compose up -d --build`.

Tested: 80/80 requests answered through Compose while a replica was killed (SIGKILL); 150/150
through a Kubernetes Service (kind) while an API pod was deleted; pushed tracks served by every
replica after their next refresh.

## Web UI

`earwig serve` (without `--api-only`) opens a page compiled into the binary:

- **Explore** — mix a reference track (from the library or any uploaded file), moods, styles, a
  description and a tempo, with presets; play the results in the page.
- **Benchmark** — `earwig eval` in the browser: precision of each weighting against random, per
  label, and what gets recommended instead. `serve --labels file.tsv` makes it leak-free.
- **Listening test** — two weightings merged and shuffled, rated blind, then revealed; a scoreboard
  of human precision. Ratings live in the store, so every replica sees them.

It binds to `127.0.0.1`; `--host 0.0.0.0` exposes the library **and its audio files**.

## How it works

| Step | Crate | Output |
|---|---|---|
| Tags | `lofty` | title, artist, album, genre, year; file name if missing |
| Decoding | `symphonia` (+ libopus) | mono PCM; damaged frames skipped, like players do |
| MIR | `stratum-dsp` (middle minute) | BPM, key, beat-grid stability, RMS energy |
| Sound embedding | log-mel (`rustfft`) → [CLAP](https://huggingface.co/laion/clap-htsat-unfused) via `ort` | 512-d vector, mean of 3 × 10 s windows |
| Auto-tags | CLAP text tower, zero-shot | 2 genres + 3 moods |
| Text embedding | `fastembed` (multilingual MiniLM) | 384-d vector of "genres, moods, themes, tags" |

The Rust log-mel front-end reproduces Hugging Face's `ClapFeatureExtractor` (cosine 0.992 with the
Python pipeline). Scoring is a weighted mean of each part, standardised (z-score) over the library
([`scoring.rs`](crates/earwig-core/src/scoring.rs)); defaults (sound 0.7, tags 0.3) come from
`earwig eval --grid`. Search is an exact SIMD scan over all cores, so weights can change on every
query:

| Tracks | Vector RAM | Query (10 cores) |
|---|---|---|
| 10,000 | 36 MB | 1.2 ms |
| 100,000 | 358 MB | 4.5 ms |
| 1,000,000 | 3.6 GB | 42 ms |

Indexing is incremental (size, mtime) and ids are hashes of the paths, so shards never collide and
ids survive re-indexing. A run that finds no audio publishes nothing, so a wrong path never empties
a library.

## Evaluation

Proxies, not listeners: how often the top-10 shares the query's label, against a random ranking.

| Dataset | Same genre @10 | Same artist @10 |
|---|---|---|
| Jamendo, 3,750 tracks, 15 styles × 250, genre not in the files | **41.5 %** vs 6.6 % random (×6.3) | 8.6 % vs 0.1 % |
| MTG-Jamendo, 1,143 tracks | **50.8 %** vs 19.4 % | 6.2 % vs 0.3 % |
| A personal library, 118 tagged tracks | | **74.5 %** vs 15.3 % |

Per style on the 3,750 tracks: from 20 % (world) to 54 % (classical), 60 % (rap) and 67 % (metal);
the confusions are rock ↔ metal and soundtrack ↔ classical. Against **Essentia**'s Discogs-EffNet
embeddings (trained on 400 Discogs styles) on the same tracks and windows: 46.3 % vs 41.5 % for
CLAP; both concatenated: 49.1 %. Search by words: styles ×4.6 over random, moods ×2.2.

<details><summary>Reproduce</summary>

```bash
# MTG-Jamendo (non-commercial research only; do not redistribute the audio)
mkdir -p data/jamendo && cd data/jamendo
B=https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data
curl -fLO $B/autotagging_genre.tsv
for n in 00 01; do curl -fLO https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/raw_30s_audio-low-$n.tar && tar -xf raw_30s_audio-low-$n.tar && rm raw_30s_audio-low-$n.tar; done
cd ../..
earwig --store data/jamendo-library index data/jamendo
earwig --store data/jamendo-library eval --by genre --labels data/jamendo/autotagging_genre.tsv --grid
```

The balanced 3,750-track corpus is built with [`scripts/jamendo-corpus.py`](scripts/jamendo-corpus.py)
(free Jamendo API key). Any external embedding can be compared the same way with
`cargo run --release -p earwig --example swap_audio`.
</details>

## As a library

| Crate | |
|---|---|
| [`earwig-core`](crates/earwig-core) | `TrackMeta`, `Index`, `Query`, scoring — no ML dependency, enough to serve |
| [`earwig-analysis`](crates/earwig-analysis) | `Analyzer` (decoding, MIR, CLAP, auto-tags, MiniLM), `indexer::update` |
| [`earwig`](crates/earwig) | the binary: index, similar, serve, eval |

```rust
use earwig_analysis::Analyzer;
use earwig_core::{scoring::{top_k, Weights}, Index};

let index = Index::load("music.index".as_ref())?;
let (_meta, query) = Analyzer::load()?.analyze_query("song.mp3".as_ref())?;
for (i, s) in top_k(&index.tracks, &query, None, &Weights::default(), 10) {
    println!("{:.3}  {}", s.score, index.tracks[i].meta.title);
}
```

## Limitations

- Relevance is measured with proxies (same genre / artist), not yet with listeners at scale.
- Mood search is weaker than style search; descriptions must be in English.
- Zero-shot auto-tags are unreliable on some styles (rap tagged "bossa nova"): your metadata wins,
  and explanations never show auto-tags.
- Tempo detection is less accurate than Essentia's; BPM has no weight by default.
- Past ~1M tracks, an approximate pre-selection (HNSW) and f16 vectors would be needed.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): checks before a PR, backing relevance changes with numbers,
and how releases work.

## License

Code: [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. Models are downloaded at
run time, not redistributed: CLAP (`laion/clap-htsat-unfused`) and
paraphrase-multilingual-MiniLM-L12-v2, both Apache-2.0.
