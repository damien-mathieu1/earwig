# syntax=docker/dockerfile:1
# earwig image: the binary + the models baked in (no download at run time).
# State lives in the library store (EARWIG_STORE: s3://bucket/prefix or a mounted directory),
# so the same image runs stateless `serve` replicas and `index` shard workers.
#
#   docker build -t earwig .
#   docker compose up            # S3 (RustFS) + 2 index shards + 2 replicas behind nginx (see docker-compose.yml)

# trixie: the prebuilt ONNX Runtime needs glibc >= 2.38 and libstdc++ from GCC 14
FROM rust:1-trixie AS build
# cmake builds the bundled libopus (Opus decoding)
RUN apt-get update && apt-get install -y --no-install-recommends cmake && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# cache mounts: rebuilds only recompile what changed (needs BuildKit, the default with buildx/compose)
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/root/.cache \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked --bin earwig && cp target/release/earwig /usr/local/bin/earwig

FROM debian:trixie-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libssl3t64 libstdc++6 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --home /nonexistent earwig
COPY --from=build /usr/local/bin/earwig /usr/local/bin/earwig
ENV EARWIG_CACHE=/models
# ~700 MB of models (CLAP audio + text, MiniLM), fetched once at build time
RUN earwig models && chmod -R a+rX /models
# a local library store for `docker run -v …:/data`; Kubernetes uses S3 (EARWIG_STORE) instead
# (`earwig models` above already created /data/library as root)
RUN mkdir -p /data && chown -R 10001:10001 /data
ENV EARWIG_STORE=/data/library
USER 10001
EXPOSE 8081
ENTRYPOINT ["earwig"]
# inside a container, 0.0.0.0 is needed for the port to be reachable; keep it behind your proxy
CMD ["serve", "--host", "0.0.0.0"]
