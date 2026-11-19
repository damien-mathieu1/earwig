#!/usr/bin/env python3
"""Build a demo corpus of *genuine* mainstream tracks from the iTunes Search API.

Apple publishes a public, unauthenticated search endpoint that returns, for every
song, the real catalogue metadata (artist, album, genre, year) and a 30-second
preview URL. 30 s is exactly the scale `earwig-analysis` works at (it averages
three 10 s windows at 1/6, 1/2 and 5/6 of the file), so previews index and play
fine — no full-length audio needed.

    python3 scripts/itunes-corpus.py --per-genre 200 --out data/genuine

    earwig --index data/genuine.bin index data/genuine
    earwig --index data/genuine.bin eval --by genre --labels data/genuine/labels.tsv
    earwig --index data/genuine.bin serve

Terms of use: this is a *local evaluation* corpus. Apple's Search API terms do
not authorise building or redistributing a corpus, and the previews stay
Apple's. Keep the files on your machine, do not ship them, do not publish the
audio. For a redistributable corpus use scripts/jamendo-corpus.py (CC) instead.
Both scripts write labels.tsv in the MTG-Jamendo format, so `eval --labels`
never leaks the genre into the text embedding.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SEARCH = "https://itunes.apple.com/search"

# bucket -> (queries sent to the API, primaryGenreName substrings accepted)
# The query set matters more than country: the French storefront carries both
# French and international catalogue, so one storefront is enough.
GENRES = {
    "rap": (["Hip-Hop/Rap", "french rap", "rap français", "rap us", "rap conscient",
             "trap", "drill", "boom bap"],
            ["hip-hop", "hip hop", "rap"]),
    "reggae": (["reggae", "dancehall", "ragga", "ska", "roots reggae", "dub", "reggae français"],
               ["reggae", "dancehall", "ragga", "ska"]),
    "rock": (["rock", "alternative rock", "indie rock", "punk rock", "grunge",
              "garage rock", "hard rock", "rock français"],
             ["rock", "alternative", "grunge", "punk", "indie"]),
    "metal": (["metal", "heavy metal", "metalcore", "death metal", "black metal", "nu metal"],
              ["metal", "hardcore"]),
    "pop": (["pop", "pop française", "synth pop", "dance pop", "variété française",
             "chanson française", "k-pop"],
            ["pop", "variété"]),
    "electronic": (["electronic", "techno", "house music", "drum and bass", "dubstep",
                    "trance", "electro", "idm", "french touch"],
                   ["electronic", "électronique", "dance", "house", "techno", "electro",
                    "trance", "drum", "dubstep"]),
    "jazz": (["jazz", "bebop", "swing jazz", "jazz fusion", "smooth jazz", "free jazz"],
             ["jazz", "swing"]),
    "classical": (["classical", "baroque", "orchestral", "piano solo", "opéra", "symphonie"],
                  ["classical", "classique", "orchestral", "baroque", "opera", "opéra"]),
    "folk": (["folk", "acoustic", "country", "singer songwriter", "americana", "bluegrass"],
             ["folk", "country", "acoustic", "singer", "americana", "bluegrass"]),
    "soul": (["soul", "funk", "r&b", "rnb", "disco", "motown", "neo soul"],
             ["soul", "r&b", "rnb", "funk", "disco", "motown"]),
    "latin": (["latin", "reggaeton", "salsa", "bossa nova", "cumbia", "bachata", "samba"],
              ["latin", "reggaeton", "salsa", "bossa", "cumbia", "bachata", "samba"]),
    "world": (["world music", "afrobeat", "afropop", "raï", "zouk", "musique du monde",
               "celtic", "bollywood"],
              ["world", "afro", "monde", "raï", "rai", "zouk", "celtic", "bollywood"]),
    "blues": (["blues", "delta blues", "gospel", "chicago blues"],
              ["blues", "gospel"]),
    "ambient": (["ambient", "chillout", "lounge", "downtempo", "new age", "drone"],
                ["ambient", "chill", "lounge", "downtempo", "new age", "drone"]),
    "soundtrack": (["soundtrack", "film score", "bande originale", "game soundtrack",
                    "musique de film"],
                   ["soundtrack", "bande originale", "score", "film", "musique de"]),
}


def search(term, country, limit=200, tries=5):
    params = {"term": term, "entity": "song", "limit": limit, "country": country}
    url = SEARCH + "?" + urllib.parse.urlencode(params)
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; earwig-corpus/1.0)"})
            with urllib.request.urlopen(req, timeout=40) as r:
                if r.status != 200:
                    raise urllib.error.HTTPError(url, r.status, "bad status", r.headers, None)
                doc = json.loads(r.read().decode("utf-8"))
            return doc.get("results") or []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == tries - 1:
                print("\n  search failed: {!r}".format(term), file=sys.stderr)
                return []
            time.sleep(5 * (attempt + 1))  # iTunes throttles hard, back off
    return []


def accepted(genre_name, allow):
    g = (genre_name or "").lower()
    return any(a in g for a in allow)


def artist_key(artist):
    """Main artist only: "Lecrae & CèJae", "X feat. Y" count as Lecrae / X.

    Same rule as `index_label` in crates/earwig/src/eval.rs, so the per-artist
    cap actually limits the artist `eval --by artist` will measure.
    """
    a = (artist or "").lower().replace(" feat. ", " feat ").replace(" ft. ", " ft ")
    for sep in (" & ", ",", " feat ", " ft ", " x "):
        a = a.split(sep)[0]
    return a.strip()


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def download(url, dest, tries=3):
    tmp = dest + ".part"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "earwig-corpus/1.0"})
            with urllib.request.urlopen(req, timeout=90) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if os.path.getsize(tmp) < 20_000:
                raise IOError("suspiciously small file")
            os.replace(tmp, dest)
            return
        except (urllib.error.URLError, IOError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def write_tags(path, track, bucket):
    meta = [
        ("title", track.get("trackName") or ""),
        ("artist", track.get("artistName") or ""),
        ("album", track.get("collectionName") or ""),
        ("genre", bucket),
        ("date", (track.get("releaseDate") or "")[:10]),
        ("comment", "itunes:{} {}".format(track.get("trackId"), track.get("trackViewUrl") or "")),
    ]
    tmp = path + ".tag.m4a"
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path]
    for k, v in meta:
        if v:
            args += ["-metadata", "{}={}".format(k, v.replace("=", "-"))]
    args += ["-c", "copy", tmp]
    subprocess.run(args, check=True)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/genuine")
    ap.add_argument("--per-genre", type=int, default=200)
    ap.add_argument("--genres", default="")
    ap.add_argument("--max-per-artist", type=int, default=4,
                    help="previews are short: keep the artist spread wide (default: 4)")
    ap.add_argument("--country", default="fr", help="storefront (default: fr)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    args = ap.parse_args()

    wanted = [g.strip() for g in args.genres.split(",") if g.strip()] or list(GENRES)
    unknown = [g for g in wanted if g not in GENRES]
    if unknown:
        raise SystemExit("unknown genre(s): {} — pick from {}".format(unknown, ", ".join(GENRES)))

    os.makedirs(args.out, exist_ok=True)
    seen, per_album, manifest, tsv = set(), {}, [], []
    print("{} buckets x {} tracks, storefront {}: ~{:.1f} GB, ~{} API calls".format(
        len(wanted), args.per_genre, args.country,
        len(wanted) * args.per_genre * 0.0008,
        sum(len(GENRES[b][0]) for b in wanted)))

    for bucket in wanted:
        queries, allow = GENRES[bucket]
        got = 0
        print("== {} — target {}".format(bucket, args.per_genre), flush=True)
        for term in queries:
            if got >= args.per_genre:
                break
            results = search(term, args.country)
            for t in results:
                if got >= args.per_genre:
                    break
                tid = t.get("trackId")
                if not tid or tid in seen:
                    continue
                if t.get("kind") != "song" or not t.get("previewUrl"):
                    continue
                if not accepted(t.get("primaryGenreName"), allow):
                    continue
                artist = (t.get("artistName") or "").strip()
                if artist and per_artist.get(artist, 0) >= args.max_per_artist:
                    continue
                seen.add(tid)
                per_artist[artist] = per_artist.get(artist, 0) + 1
                got += 1
                dest = os.path.join(args.out, bucket, "{}.m4a".format(tid))
                if args.dry_run:
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                try:
                    if not os.path.exists(dest):
                        if free_gb(args.out) < args.min_free_gb:
                            raise SystemExit("stopping: less than {:.0f} GB free".format(
                                args.min_free_gb))
                        download(t["previewUrl"], dest)
                        write_tags(dest, t, bucket)
                except Exception as e:
                    print("\n  skip {}: {}".format(tid, e), file=sys.stderr)
                    seen.discard(tid)
                    per_artist[artist] -= 1
                    got -= 1
                    continue
                manifest.append({
                    "track_id": tid, "title": t.get("trackName"),
                    "artist": artist, "album": t.get("collectionName"),
                    "genre": bucket, "itunes_genre": t.get("primaryGenreName"),
                    "release": (t.get("releaseDate") or "")[:10],
                    "duration_ms": t.get("trackTimeMillis"), "file": os.path.relpath(dest),
                    "itunes_url": t.get("trackViewUrl"),
                })
                tsv.append([str(tid), str(t.get("artistId") or ""),
                            str(t.get("collectionId") or ""), dest,
                            str(t.get("trackTimeMillis") or ""), "genre---" + bucket])
            sys.stdout.write("\r  {} {}/{}  ".format(bucket, got, args.per_genre))
            sys.stdout.flush()
            time.sleep(3)  # stay under the ~20 requests/minute throttle
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
    print("\nnext:")
    print("  cargo run --release -p earwig -- --index {0}.bin index {0}".format(args.out))
    print("  cargo run --release -p earwig -- --index {0}.bin eval --by genre "
          "--labels {0}/labels.tsv --grid".format(args.out))
    print("  cargo run --release -p earwig -- --index {0}.bin serve".format(args.out))


if __name__ == "__main__":
    main()
