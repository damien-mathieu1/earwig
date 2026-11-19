//! earwig — content-based music recommendation.
//!
//!   earwig index ~/Music            analyse a folder or s3://bucket/prefix (incremental, shardable)
//!   earwig similar song.mp3         10 closest tracks to a file (or to an indexed track id)
//!   earwig serve                    web UI + JSON API on 127.0.0.1:8081 (stateless replica)
//!   earwig eval                     how often the top-k shares the source's artist / genre
//!
//! Every command works on a library store (`--store`, a directory or s3://bucket/prefix)
//! holding immutable index snapshots; see `storage.rs` and `library.rs`.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use earwig::{library, storage::Store};
use earwig_core::scoring::{top_k, Weights};

#[derive(Parser)]
#[command(
    version,
    about = "Content-based music recommendation (CLAP audio + text embeddings + BPM)"
)]
struct Cli {
    /// Library store: a local directory or s3://bucket/prefix (credentials from AWS_* variables)
    #[arg(
        long,
        global = true,
        env = "EARWIG_STORE",
        default_value = "data/library"
    )]
    store: String,
    /// Log format: `text` for people, `json` (one object per line) for log collectors
    #[arg(long, global = true, env = "EARWIG_LOG_FORMAT", default_value = "text", value_parser = ["text", "json"])]
    log_format: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Analyse the audio files of MEDIA and publish a new snapshot (only new or modified files are analysed)
    Index {
        /// Local directory or s3://bucket/prefix holding the audio files
        #[arg(env = "EARWIG_MEDIA")]
        media: String,
        /// Number of shard workers indexing in parallel
        #[arg(long, env = "EARWIG_SHARDS", default_value_t = 1)]
        shards: usize,
        /// This worker's shard, 0-based (defaults to a Kubernetes Indexed Job's JOB_COMPLETION_INDEX)
        #[arg(long, env = "EARWIG_SHARD")]
        shard: Option<usize>,
        /// Run id shared by the shard workers of one indexing run (required with --shards > 1)
        #[arg(long, env = "EARWIG_RUN")]
        run: Option<String>,
        /// JSON catalogue overriding the metadata of files named <id>.wav (demo dataset)
        #[arg(long)]
        catalog: Option<PathBuf>,
    },
    /// Merge the shards of a run and publish it (normally done by the last shard worker)
    Merge {
        #[arg(long, env = "EARWIG_RUN")]
        run: String,
        #[arg(long, env = "EARWIG_SHARDS")]
        shards: usize,
    },
    /// Print the tracks closest to an audio file or to an indexed track id
    Similar {
        /// Path to an audio file, or the id of an indexed track
        track: String,
        #[arg(short, default_value_t = 10)]
        k: usize,
        #[command(flatten)]
        weights: WeightArgs,
    },
    /// Serve the web UI and the JSON API (stateless: run as many replicas as you like)
    Serve {
        #[arg(long, env = "PORT", default_value_t = 8081)]
        port: u16,
        /// Interface to bind; 0.0.0.0 exposes your library (and its files) to the network
        #[arg(long, env = "EARWIG_HOST", default_value = "127.0.0.1")]
        host: String,
        /// MTG-Jamendo style TSV(s), local or s3://, giving the ground truth for the Benchmark tab
        #[arg(long, env = "EARWIG_LABELS", value_delimiter = ',')]
        labels: Vec<String>,
        /// Seconds between checks for a newer snapshot
        #[arg(long, env = "EARWIG_REFRESH", default_value_t = 30)]
        refresh: u64,
        /// Require `Authorization: Bearer <token>` on /api/*; several tokens comma-separated, to
        /// rotate without downtime (prefer the env variable to a flag)
        #[arg(long, env = "EARWIG_API_TOKEN", hide_env_values = true)]
        api_token: Option<String>,
        /// all, api (recommendations only, no analysis models) or indexer (pushed tracks and
        /// uploads): run api and indexer replicas apart so analysis never slows recommendations
        #[arg(long, env = "EARWIG_ROLE", value_enum, default_value_t = earwig::server::Role::All)]
        role: earwig::server::Role,
        /// Requests in flight on /api before answering 503 (Retry-After: 1) instead of queueing
        #[arg(long, env = "EARWIG_MAX_CONCURRENT", default_value_t = 256)]
        max_concurrent: usize,
        /// Only serve /api/v1 and the health probes (no web UI, benchmark, ratings or uploads page)
        #[arg(long, env = "EARWIG_API_ONLY")]
        api_only: bool,
        /// Media folder or s3:// URL, overriding the one recorded in the snapshot (audio playback,
        /// tracks pushed with a `media_key`, including before any snapshot exists)
        #[arg(long, env = "EARWIG_MEDIA")]
        media: Option<String>,
    },
    /// Relevance proxy: share of the top-k with the same label as the source, per weighting
    Eval {
        /// Ground truth: artist / genre / subgenre from the index, or artist / genre / mood from --labels
        #[arg(long, default_value = "artist", value_parser = ["artist", "genre", "subgenre", "mood"])]
        by: String,
        /// MTG-Jamendo style TSV(s), local or s3://, matched on the file name (keeps labels out of the text embedding)
        #[arg(long, value_delimiter = ',')]
        labels: Vec<String>,
        #[arg(short, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 20.0)]
        bpm_tolerance: f32,
        /// Also try every weighting on a 0.1 grid and print the best ones
        #[arg(long)]
        grid: bool,
    },
    /// Download the models into the cache (e.g. while building a container image)
    Models,
}

#[derive(Args)]
struct WeightArgs {
    #[arg(long, default_value_t = 0.7)]
    w_audio: f32,
    #[arg(long, default_value_t = 0.3)]
    w_text: f32,
    #[arg(long, default_value_t = 0.0)]
    w_bpm: f32,
    #[arg(long, default_value_t = 20.0)]
    bpm_tolerance: f32,
}

async fn similar(store: &Store, track: &str, k: usize, w: Weights) -> anyhow::Result<()> {
    let index = library::require_latest(store).await?;
    let path = std::path::Path::new(track);
    let top = if path.is_file() {
        let path = path.to_path_buf();
        let (meta, q) = tokio::task::spawn_blocking(move || {
            earwig_analysis::Analyzer::load()?.analyze_query(&path)
        })
        .await??;
        println!(
            "{} — {}  [{:.0} bpm, {}, {} / {}]\n",
            meta.title,
            meta.artist,
            meta.bpm,
            meta.key,
            meta.auto_genres.join(", "),
            meta.auto_moods.join(", ")
        );
        top_k(&index.tracks, &q, None, &w, k)
    } else {
        let id: i64 = track
            .parse()
            .map_err(|_| anyhow::anyhow!("{track} is neither a file nor a track id"))?;
        let i = index
            .tracks
            .iter()
            .position(|t| t.meta.id == id)
            .ok_or_else(|| anyhow::anyhow!("no track with id {id}"))?;
        let m = &index.tracks[i].meta;
        println!(
            "{} — {}  [{:.0} bpm, {}]\n",
            m.title, m.artist, m.bpm, m.key
        );
        top_k(&index.tracks, &index.tracks[i], Some(i), &w, k)
    };
    println!(
        "{:>3}  {:>5}  {:>5}  {:>5}  {:>5}  track",
        "#", "score", "audio", "text", "Δbpm"
    );
    for (rank, (i, s)) in top.iter().enumerate() {
        let m = &index.tracks[*i].meta;
        println!(
            "{:>3}  {:.3}  {:.3}  {:.3}  {:>5.1}  {} — {}  ({})",
            rank + 1,
            s.score,
            s.audio,
            s.text,
            s.bpm_delta,
            m.title,
            m.artist,
            m.path
        );
    }
    Ok(())
}

async fn read_labels(urls: &[String]) -> anyhow::Result<Vec<String>> {
    let mut out = Vec::new();
    for u in urls {
        out.push(String::from_utf8(earwig::storage::read_any(u).await?)?);
    }
    Ok(out)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    earwig::telemetry::init_logging(cli.log_format == "json");
    let store = Store::open(&cli.store)?;
    match cli.cmd {
        Cmd::Index {
            media,
            shards,
            shard,
            run,
            catalog,
        } => {
            let shard = match shard {
                Some(s) => s,
                None => std::env::var("JOB_COMPLETION_INDEX")
                    .ok()
                    .map(|v| v.parse())
                    .transpose()?
                    .unwrap_or(0),
            };
            anyhow::ensure!(shards == 1 || run.is_some(), "--run (or EARWIG_RUN) is required with --shards > 1, so the workers agree on one run");
            let run = run.unwrap_or_else(library::new_run_id);
            let catalog = match catalog {
                Some(p) => earwig::catalog::load_catalog(&p)?,
                None => vec![],
            };
            let job = library::IndexJob {
                media,
                shard,
                shards,
                run,
            };
            match library::index(&store, &job, library::catalog_editor(catalog)).await? {
                Some(key) => println!("published {key} in {}", store.url),
                None => println!(
                    "shard {} written; the last shard of run {} will publish",
                    job.shard, job.run
                ),
            }
            Ok(())
        }
        Cmd::Merge { run, shards } => match library::merge(&store, &run, shards).await? {
            Some(key) => {
                println!("published {key}");
                Ok(())
            }
            None => anyhow::bail!("run {run}: not all {shards} shards are written yet"),
        },
        Cmd::Similar {
            track,
            k,
            weights: w,
        } => {
            let w = Weights {
                w_audio: w.w_audio,
                w_text: w.w_text,
                w_bpm: w.w_bpm,
                bpm_tolerance: w.bpm_tolerance,
            };
            similar(&store, &track, k, w).await
        }
        Cmd::Serve {
            port,
            host,
            labels,
            refresh,
            api_token,
            role,
            max_concurrent,
            api_only,
            media,
        } => {
            let cfg = earwig::server::ServeConfig {
                host,
                port,
                label_tsvs: read_labels(&labels).await?,
                refresh_secs: refresh,
                api_token,
                role,
                max_concurrent,
                api_only,
                media,
            };
            earwig::server::serve(store, cfg).await
        }
        Cmd::Eval {
            by,
            labels,
            k,
            bpm_tolerance,
            grid,
        } => {
            let index = library::require_latest(&store).await?;
            let tsvs = read_labels(&labels).await?;
            tokio::task::block_in_place(|| {
                earwig::eval::run(&index, &by, &tsvs, k, bpm_tolerance, grid)
            })
        }
        Cmd::Models => {
            tokio::task::spawn_blocking(earwig_analysis::Analyzer::load).await??;
            println!(
                "models cached in {}",
                earwig_analysis::cache_dir().display()
            );
            Ok(())
        }
    }
}
