# Contributing

Thanks for helping! Bug reports, evaluation results on your own library and pull requests are all welcome.

## Before a pull request

```bash
cargo fmt --all
```

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

```bash
cargo test --workspace --release
```

Tests do not download models. To try the full pipeline without your own music, use the demo dataset
(see the README).

## Changing the scoring or the analysis

Back relevance changes with numbers: run `earwig eval` (on your library and, if you can, on the
MTG-Jamendo sample described in the README) before and after, and put both in the PR.

## Commits

[Conventional commits](https://www.conventionalcommits.org) (`feat:`, `fix:`, `docs:`, `perf:`…):
the release notes are generated from them ([`cliff.toml`](cliff.toml)).

## Releasing

Versions are CalVer `vYYYY.M.N` (N = release number in the month). To release, go to
**Actions → Create Release → Run workflow** on `main`. It:

1. runs the CI (fmt, clippy, tests on Linux and macOS; nothing happens if they fail),
2. computes the next version, bumps `Cargo.toml` (the three crates and their exact pins),
   commits `chore(release): vX` to `main` and tags it,
3. creates the GitHub Release with a changelog built from the conventional commits,
4. builds the binaries (Linux x86_64 / arm64, macOS arm64) and attaches them to the release,
5. builds the container image natively for amd64 and arm64 and pushes
   `ghcr.io/damien-mathieu1/earwig:vX` and `:latest`,
6. publishes `earwig-core`, `earwig-analysis` and `earwig` to crates.io (needs the
   `CARGO_REGISTRY_TOKEN` repo secret),
7. updates the formula in `damien-mathieu1/homebrew-tap` (needs the `TAP_DEPLOY_KEY` repo secret:
   private half of a write deploy key on the tap).

## License

By contributing you agree that your work is dual-licensed under MIT and Apache-2.0, like the project.
