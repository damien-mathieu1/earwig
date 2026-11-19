#!/usr/bin/env python3
"""Build a genre-balanced earwig test corpus from the Jamendo API.

Downloads full-length, Creative Commons tracks (only those whose author allows
download), writes ID3 tags (title, artist, album, date) so `earwig index` picks the
metadata up, and emits an MTG-Jamendo style TSV for `earwig eval --labels`.

The genre is NOT written to the files by default: earwig embeds the genre tag in its
text vector, so a genre tag equal to the eval label would hand it the answer (text-only
~100 %, `--grid` pushing all weight to text). Without it the text vector comes from CLAP
auto-tags and `eval --by genre` is leak-free. `--genre-tag` writes it anyway, to measure
the "real library with good tags" case — never read genre results from such a corpus as
the algorithm's quality.

Get a free client_id at https://devportal.jamendo.com (an App, "read" methods).

    export JAMENDO_CLIENT_ID=xxxx
    python3 scripts/jamendo-corpus.py --per-genre 250 --out data/corpus

    earwig --store data/corpus-library index data/corpus
    earwig --store data/corpus-library eval --by genre  --labels data/corpus/labels.tsv --grid   # leak-free
    earwig --store data/corpus-library eval --by artist --labels data/corpus/labels.tsv
    earwig --store data/corpus-library serve

Licence: Jamendo tracks are CC (per-track, see corpus/manifest.jsonl). Non-commercial
research use only; do not redistribute the audio. The manifest keeps the licence URL
and artist name of every track for attribution.
"""

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.jamendo.com/v3.0/tracks/"

# Jamendo tag -> the single genre label written in the ID3 tag and in the TSV,
# so `eval --by genre` answers "do the top-10 stay inside the genre bucket?".
GENRES = {
    "rap": ["rap", "hiphop", "trap", "gangsta"],
    "reggae": ["reggae", "dub", "ska", "dancehall", "roots"],
    "rock": ["rock", "alternative", "indie-rock", "punk", "grunge"],
    "metal": ["metal", "hardcore", "metalcore"],
    "pop": ["pop", "synthpop", "eurodance"],
    "electronic": ["electronic", "techno", "house", "trance", "drum-and-bass", "dubstep", "idm"],
    "jazz": ["jazz", "swing", "bebop", "fusion"],
    "classical": ["classical", "orchestral", "baroque"],
    "folk": ["folk", "country", "singer-songwriter"],
    "soul": ["soul", "funk", "rnb", "disco", "motown"],
    "latin": ["latin", "salsa", "bossa", "samba", "cumbia"],
    "world": ["world", "afrobeat", "oriental", "balkan"],
    "blues": ["blues", "gospel"],
    "ambient": ["ambient", "chillout", "lounge", "downtempo"],
    "soundtrack": ["soundtrack"],
}

KEEP = ("id", "name", "artist_id", "artist_name", "album_id", "album_name",
        "duration", "releasedate", "audio", "audiodownload",
        "audiodownload_allowed", "image", "shareurl", "musicinfo", "license_ccurl")


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def api(client_id, params, tries=4):
    query = dict(params, client_id=client_id, format="json")
    url = API + "?" + urllib.parse.urlencode(query, doseq=True)
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "earwig-corpus/1.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                doc = json.loads(r.read().decode("utf-8"))
            head = doc.get("headers", {})
            if head.get("status") != "success":
                raise SystemExit("Jamendo error: {} (code {})".format(
                    head.get("error_message"), head.get("code")))
            return doc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == tries - 1:
                raise SystemExit("Jamendo API unreachable: {}".format(e))
            time.sleep(2 * (attempt + 1))


def tracks_for_tag(client_id, tag, want, artist_cap, seen, per_artist,
                   min_dur, max_dur, lang, want_lyrics):
    """Yield acceptable tracks for one Jamendo tag, paging 200 by 200."""
    offset = 0
    taken = 0
    while taken < want and offset < 4000:
        params = {
            "limit": 200,
            "offset": offset,
            "tags": tag,
            "audioformat": "mp32",
            "audiodlformat": "mp32",
            "durationbetween": "{}_{}".format(min_dur, max_dur),
            "include": "musicinfo licenses" + (" lyrics" if want_lyrics else ""),  # space-separated; urlencode turns it into +
        }
        if lang:
            params["lang"] = lang
        # the API intermittently answers an empty "success" page: retry before calling it the end
        for attempt in range(4):
            results = api(client_id, params).get("results") or []
            if results:
                break
            time.sleep(0.5 * (attempt + 1))
        if not results:
            return
        for t in results:
            if taken >= want:
                return
            tid = t.get("id")
            if not tid or tid in seen:
                continue
            if not t.get("audiodownload_allowed") or not t.get("audiodownload"):
                continue
            artist = (t.get("artist_name") or "").strip()
            if artist and per_artist.get(artist, 0) >= artist_cap:
                continue
            seen.add(tid)
            per_artist[artist] = per_artist.get(artist, 0) + 1
            taken += 1
            yield {k: t.get(k) for k in KEEP if k in t}
        offset += 200
        time.sleep(0.2)


def download(url, dest, tries=3):
    tmp = dest + ".part"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "earwig-corpus/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if os.path.getsize(tmp) < 20_000:
                raise IOError("suspiciously small file")
            os.replace(tmp, dest)
            return
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def write_tags(path, track, genre=None):
    """ffmpeg -c copy: no re-encode, just ID3v2.3 frames lofty can read."""
    meta = [
        ("title", html.unescape(track.get("name") or "")),
        ("artist", html.unescape(track.get("artist_name") or "")),
        ("album", html.unescape(track.get("album_name") or "")),
        ("genre", genre),
        ("date", (track.get("releasedate") or "")[:10]),
        ("comment", "jamendo:{} {}".format(track.get("id"), track.get("license_ccurl") or "")),
    ]
    tmp = path + ".tag.mp3"
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path]
    for k, v in meta:
        if v:
            args += ["-metadata", "{}={}".format(k, v.replace("=", "-"))]
    if genre is None:
        # Jamendo's MP3s already carry a genre tag ("trap, hiphop"): erase it, or the label leaks anyway
        args += ["-metadata", "genre="]
    args += ["-c", "copy", tmp]
    subprocess.run(args, check=True)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/corpus", help="corpus root (default: data/corpus)")
    ap.add_argument("--per-genre", type=int, default=200,
                    help="tracks to keep per genre bucket (default: 200)")
    ap.add_argument("--genres", default="", help="comma-separated subset of: " + ", ".join(GENRES))
    ap.add_argument("--max-per-artist", type=int, default=6)
    ap.add_argument("--min-duration", type=int, default=90)
    ap.add_argument("--max-duration", type=int, default=420)
    ap.add_argument("--lang", default="", help="lyrics language filter, e.g. fr (French rap)")
    ap.add_argument("--lyrics", action="store_true",
                    help="also save Jamendo lyrics sidecars (not used by earwig today)")
    ap.add_argument("--genre-tag", action="store_true",
                    help="also write the genre bucket to ID3 (leaks the eval label into earwig's text vector)")
    ap.add_argument("--dry-run", action="store_true", help="count matches, download nothing")
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    args = ap.parse_args()

    client_id = os.environ.get("JAMENDO_CLIENT_ID", "").strip()
    if not client_id:
        raise SystemExit("Set JAMENDO_CLIENT_ID (free key: https://devportal.jamendo.com)")

    wanted = [g.strip() for g in args.genres.split(",") if g.strip()] or list(GENRES)
    unknown = [g for g in wanted if g not in GENRES]
    if unknown:
        raise SystemExit("unknown genre(s): {} — pick from {}".format(unknown, ", ".join(GENRES)))

    os.makedirs(args.out, exist_ok=True)
    seen, per_artist, manifest, tsv = set(), {}, [], []

    for bucket in wanted:
        tags = GENRES[bucket]
        per_tag = max(1, args.per_genre // len(tags))
        print("== {} ({}) — target {} ({} per tag)".format(
            bucket, ",".join(tags), args.per_genre, per_tag), flush=True)
        got = 0
        # pass 1: a fair share per tag; pass 2: whatever is left goes to the tags that still have tracks
        for tag, want in [(t, per_tag) for t in tags] + [(t, None) for t in tags]:
            if got >= args.per_genre:
                break
            want = min(want or args.per_genre, args.per_genre - got)
            genre_dir = os.path.join(args.out, bucket)
            os.makedirs(genre_dir, exist_ok=True)
            try:
                for tr in tracks_for_tag(client_id, tag, want, args.max_per_artist, seen,
                                         per_artist, args.min_duration, args.max_duration,
                                         args.lang, args.lyrics):
                    dest = os.path.join(genre_dir, "{}.mp3".format(tr["id"]))
                    if args.dry_run:
                        got += 1
                        continue
                    if not os.path.exists(dest):
                        if free_gb(args.out) < args.min_free_gb:
                            raise SystemExit("stopping: less than {:.0f} GB free".format(
                                args.min_free_gb))
                        download(tr["audiodownload"], dest)
                        try:
                            write_tags(dest, tr, bucket if args.genre_tag else None)
                        except Exception:
                            # an untaggable file keeps Jamendo's own genre tag (a label leak): drop it
                            os.remove(dest)
                            raise
                        time.sleep(0.2)
                    got += 1
                    info = tr.get("musicinfo") or {}
                    manifest.append({
                        "id": tr["id"], "title": html.unescape(tr.get("name") or ""), "artist": html.unescape(tr.get("artist_name") or ""),
                        "album": tr.get("album_name"), "genre": bucket, "tag": tag,
                        "duration": tr.get("duration"), "releasedate": tr.get("releasedate"),
                        "jamendo_url": tr.get("shareurl"), "license": tr.get("license_ccurl"),
                        "file": os.path.relpath(dest),
                    })
                    if args.lyrics and info.get("lyrics"):
                        ldir = os.path.join(args.out, "lyrics")
                        os.makedirs(ldir, exist_ok=True)
                        with open(os.path.join(ldir, "{}.txt".format(tr["id"])), "w") as f:
                            f.write(info["lyrics"])
                    tsv.append([str(tr["id"]), str(tr.get("artist_id") or ""),
                                str(tr.get("album_id") or ""), dest,
                                str(tr.get("duration") or ""), "genre---" + bucket])
                    sys.stdout.write("\r  {} {}/{}  ".format(bucket, got, args.per_genre))
                    sys.stdout.flush()
            except SystemExit:
                raise
            except Exception as e:  # a dead track must not kill the run
                print("\n  skip {}: {}".format(tag, e), file=sys.stderr)
        print("\r  {}: {} tracks".format(bucket, got))

    if args.dry_run:
        print("\ndry run: {} tracks match, nothing written".format(len(seen)))
        return

    with open(os.path.join(args.out, "manifest.jsonl"), "w") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "labels.tsv"), "w") as f:
        f.write("TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n")
        for row in tsv:
            f.write("\t".join(row) + "\n")

    print("\n{} tracks, max {} per artist -> {}".format(
        len(manifest), args.max_per_artist, args.out))
    print("labels: {0}/labels.tsv  attribution: {0}/manifest.jsonl".format(args.out))
    print("\nnext:")
    print("  cargo run --release -p earwig -- --store {0}-library index {0}".format(args.out))
    print("  cargo run --release -p earwig -- --store {0}-library eval --by genre "
          "--labels {0}/labels.tsv --grid".format(args.out))
    print("  cargo run --release -p earwig -- --store {0}-library serve --labels {0}/labels.tsv".format(args.out))


if __name__ == "__main__":
    main()
