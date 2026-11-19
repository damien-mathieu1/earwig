//! Relevance proxy without manual annotation (`earwig eval`, and the Benchmark tab of the UI).
//!
//! For every labelled track that shares a label with another one, precision@k = share of its
//! top-k that shares at least one label with it, averaged over queries and compared with a
//! random ranking. Labels come from the index (artist / genre / subgenre) or, to keep them out
//! of the text embedding, from MTG-Jamendo style TSVs (`--labels`).

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
};

use earwig_core::{
    scoring::{combine, hybrid, Weights},
    Track, TrackMeta,
};
use rayon::prelude::*;
use serde::Serialize;

/// "14/214.mp3" and ".../214.low.mp3" (MTG-Jamendo audio-low) both map to "214".
pub fn file_stem(path: &str) -> String {
    let name = Path::new(path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    name.split('.').next().unwrap_or_default().to_string()
}

/// MTG-Jamendo TSV: TRACK_ID ARTIST_ID ALBUM_ID PATH DURATION TAG… ("genre---rock", "mood/theme---happy").
fn load_tsv_labels(tsvs: &[String], by: &str) -> anyhow::Result<HashMap<String, HashSet<String>>> {
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for tsv in tsvs {
        for line in tsv.lines().skip(1) {
            let cols: Vec<&str> = line.split('\t').collect();
            anyhow::ensure!(cols.len() >= 6, "not an MTG-Jamendo TSV line: {line}");
            let entry = out.entry(file_stem(cols[3])).or_default();
            match by {
                "artist" => {
                    entry.insert(cols[1].to_string());
                }
                "genre" => entry.extend(
                    cols[5..]
                        .iter()
                        .filter_map(|t| t.strip_prefix("genre---"))
                        .map(String::from),
                ),
                "mood" => entry.extend(
                    cols[5..]
                        .iter()
                        .filter_map(|t| t.strip_prefix("mood/theme---"))
                        .map(String::from),
                ),
                other => anyhow::bail!(
                    "--by {other} is not available with --labels (artist, genre, mood)"
                ),
            }
        }
    }
    Ok(out)
}

fn index_label(m: &TrackMeta, by: &str) -> HashSet<String> {
    let l = match by {
        "genre" => m.genre.to_lowercase(),
        "subgenre" => m.subgenre.to_lowercase(),
        // main artist only: "Josman & Laylow" / "X feat. Y" count as Josman / X
        _ => m
            .artist
            .to_lowercase()
            .split([',', '&'])
            .next()
            .unwrap_or("")
            .split(" feat")
            .next()
            .unwrap_or("")
            .trim()
            .to_string(),
    };
    if l.is_empty() || l.contains("unknown") {
        HashSet::new()
    } else {
        HashSet::from([l])
    }
}

/// The ground truth of every track of `tracks` (empty set = not labelled). `tsvs` are the
/// contents of label files; without any, labels come from the index's own fields.
pub fn track_labels(
    tracks: &[Track],
    by: &str,
    tsvs: &[String],
) -> anyhow::Result<Vec<HashSet<String>>> {
    if tsvs.is_empty() {
        return Ok(tracks.iter().map(|t| index_label(&t.meta, by)).collect());
    }
    let tsv = load_tsv_labels(tsvs, by)?;
    Ok(tracks
        .iter()
        .map(|t| {
            tsv.get(&file_stem(&t.meta.path))
                .cloned()
                .unwrap_or_default()
        })
        .collect())
}

#[derive(Debug, Serialize)]
pub struct WeightingRow {
    pub name: String,
    pub w_audio: f32,
    pub w_text: f32,
    pub w_bpm: f32,
    pub precision: f32,
    /// precision / random
    pub lift: f32,
}

#[derive(Debug, Serialize)]
pub struct LabelRow {
    pub label: String,
    pub queries: usize,
    pub precision: f32,
    pub random: f32,
    /// share of each label among this label's top-k, most frequent first (default weighting)
    pub recommended: Vec<(String, f32)>,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub by: String,
    pub k: usize,
    pub labelled: usize,
    pub queries: usize,
    pub random: f32,
    pub weightings: Vec<WeightingRow>,
    /// best weightings of the 0.1 grid (empty unless asked)
    pub grid: Vec<WeightingRow>,
    pub per_label: Vec<LabelRow>,
}

/// Precision@k of several weightings over the labelled tracks; per-label detail for the default one.
pub fn evaluate(
    all: &[Track],
    all_labels: &[HashSet<String>],
    by: &str,
    k: usize,
    bpm_tolerance: f32,
    grid: bool,
) -> anyhow::Result<Report> {
    // only labelled tracks take part, as queries and as candidates
    let (tracks, labels): (Vec<&Track>, Vec<&HashSet<String>>) = all
        .iter()
        .zip(all_labels)
        .filter(|(_, l)| !l.is_empty())
        .unzip();
    let n = tracks.len();
    let shares = |i: usize, j: usize| !labels[i].is_disjoint(labels[j]);
    let queries: Vec<usize> = (0..n)
        .filter(|&i| (0..n).any(|j| j != i && shares(i, j)))
        .collect();
    anyhow::ensure!(
        !queries.is_empty(),
        "no two tracks share a {by} label: nothing to evaluate"
    );
    let random_of =
        |i: usize| (0..n).filter(|&j| j != i && shares(i, j)).count() as f32 / (n - 1) as f32;
    let random = queries.iter().map(|&i| random_of(i)).sum::<f32>() / queries.len() as f32;

    // sub-scores computed once per (query, candidate); every weighting just re-combines them
    let probe = Weights {
        w_audio: 1.0,
        w_text: 1.0,
        w_bpm: 1.0,
        bpm_tolerance,
    };
    let subs: Vec<Vec<[f32; 3]>> = queries
        .par_iter()
        .map(|&i| {
            (0..n)
                .map(|j| {
                    let s = hybrid(tracks[i], tracks[j], &probe);
                    [s.audio, s.text, s.bpm_score]
                })
                .collect()
        })
        .collect();
    let top = |q: usize, w: &Weights| -> Vec<usize> {
        let i = queries[q];
        let mut scored: Vec<(f32, usize)> = subs[q]
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(j, &[a, t, b])| (combine(w, a, t, b), j))
            .collect();
        let kk = k.min(scored.len());
        scored.select_nth_unstable_by(kk - 1, |x, y| y.0.total_cmp(&x.0));
        scored[..kk].iter().map(|&(_, j)| j).collect()
    };
    let hit_rate = |q: usize, t: &[usize]| {
        t.iter().filter(|&&j| shares(queries[q], j)).count() as f32 / t.len() as f32
    };
    let precision = |w: &Weights| {
        (0..queries.len())
            .into_par_iter()
            .map(|q| hit_rate(q, &top(q, w)))
            .sum::<f32>()
            / queries.len() as f32
    };
    let row = |name: &str, w: Weights| {
        let p = precision(&w);
        WeightingRow {
            name: name.into(),
            w_audio: w.w_audio,
            w_text: w.w_text,
            w_bpm: w.w_bpm,
            precision: p,
            lift: p / random,
        }
    };
    let w = |a, t, b| Weights {
        w_audio: a,
        w_text: t,
        w_bpm: b,
        bpm_tolerance,
    };
    let default = Weights {
        bpm_tolerance,
        ..Weights::default()
    };
    let weightings = vec![
        row("audio only", w(1.0, 0.0, 0.0)),
        row("text only", w(0.0, 1.0, 0.0)),
        row("BPM only", w(0.0, 0.0, 1.0)),
        row("default", default),
    ];

    let mut grid_rows = Vec::new();
    if grid {
        grid_rows = (0..=10)
            .flat_map(|a| (0..=10 - a).map(move |t| (a, t)))
            .map(|(a, t)| {
                row(
                    "grid",
                    w(a as f32 / 10.0, t as f32 / 10.0, (10 - a - t) as f32 / 10.0),
                )
            })
            .collect();
        grid_rows.sort_by(|x, y| y.precision.total_cmp(&x.precision));
        grid_rows.truncate(5);
    }

    // per label, with the default weighting: precision, random baseline, what gets recommended
    let tops: Vec<Vec<usize>> = (0..queries.len())
        .into_par_iter()
        .map(|q| top(q, &default))
        .collect();
    #[derive(Default)]
    struct Acc<'a> {
        queries: usize,
        precision: f32,
        random: f32,
        recommended: HashMap<&'a str, usize>,
        slots: usize,
    }
    let mut acc: BTreeMap<&str, Acc> = BTreeMap::new();
    for (q, t) in tops.iter().enumerate() {
        let i = queries[q];
        for l in labels[i] {
            let e = acc.entry(l.as_str()).or_default();
            e.queries += 1;
            e.precision += hit_rate(q, t);
            e.random += random_of(i);
            for &j in t {
                for lj in labels[j] {
                    *e.recommended.entry(lj.as_str()).or_default() += 1;
                }
                e.slots += 1;
            }
        }
    }
    let mut per_label: Vec<LabelRow> = acc
        .into_iter()
        .filter(|(_, e)| e.queries >= 2)
        .map(|(label, e)| {
            let (nq, p, r, slots) = (e.queries, e.precision, e.random, e.slots);
            let mut recommended: Vec<(String, f32)> = e
                .recommended
                .into_iter()
                .map(|(l, c)| (l.to_string(), c as f32 / slots.max(1) as f32))
                .collect();
            recommended.sort_by(|a, b| b.1.total_cmp(&a.1));
            recommended.truncate(5);
            LabelRow {
                label: label.to_string(),
                queries: nq,
                precision: p / nq as f32,
                random: r / nq as f32,
                recommended,
            }
        })
        .collect();
    per_label.sort_by_key(|l| std::cmp::Reverse(l.queries));

    Ok(Report {
        by: by.into(),
        k,
        labelled: n,
        queries: queries.len(),
        random,
        weightings,
        grid: grid_rows,
        per_label,
    })
}

/// CLI front-end: prints the report.
pub fn run(
    index: &earwig_core::Index,
    by: &str,
    tsvs: &[String],
    k: usize,
    bpm_tolerance: f32,
    grid: bool,
) -> anyhow::Result<()> {
    let labels = track_labels(&index.tracks, by, tsvs)?;
    let r = evaluate(&index.tracks, &labels, by, k, bpm_tolerance, grid)?;
    println!(
        "{} labelled tracks, {} queries, same-{by}@{k} — random ranking {:.1} %\n",
        r.labelled,
        r.queries,
        100.0 * r.random
    );
    let print = |x: &WeightingRow| {
        println!(
            "  {:<11} a={:.2} t={:.2} b={:.2}   {:5.1} %  (×{:.1})",
            x.name,
            x.w_audio,
            x.w_text,
            x.w_bpm,
            100.0 * x.precision,
            x.lift
        )
    };
    r.weightings.iter().for_each(print);
    if grid {
        println!("\n  best of 66 weightings (step 0.1):");
        r.grid.iter().for_each(print);
    }
    println!("\n  per {by} (default weighting, top 15 by size):");
    for l in r.per_label.iter().take(15) {
        let rec: Vec<String> = l
            .recommended
            .iter()
            .take(3)
            .map(|(x, s)| format!("{x} {:.0}%", 100.0 * s))
            .collect();
        println!(
            "  {:<18} n={:<4} {:5.1} %  (random {:4.1} %)  → {}",
            l.label,
            l.queries,
            100.0 * l.precision,
            100.0 * l.random,
            rec.join(", ")
        );
    }
    Ok(())
}
