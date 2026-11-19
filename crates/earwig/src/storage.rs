//! Shared state lives in an object store (S3, MinIO, R2, GCS-interop… or a plain local
//! directory), so `earwig serve` replicas are stateless and indexing can run anywhere.
//!
//! Library layout under the store root:
//!
//! ```text
//! latest                        key of the current snapshot (one line)
//! snapshots/<run>.bin           immutable index snapshots
//! shards/<run>/<i>-of-<n>.bin   partial indexes written by shard workers
//! ratings/<time>-<rand>.json    listening-test ratings, one object each (no write conflicts)
//! ```
//!
//! S3 credentials and endpoint come from the usual `AWS_*` environment variables
//! (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ENDPOINT`, `AWS_ALLOW_HTTP`).

use std::sync::Arc;

use anyhow::Context;
use futures::TryStreamExt;
use object_store::{
    aws::AmazonS3Builder, local::LocalFileSystem, path::Path as KeyPath, ObjectStore,
    ObjectStoreExt, PutPayload,
};

#[derive(Clone, Debug)]
pub struct Store {
    os: Arc<dyn ObjectStore>,
    root: KeyPath,
    /// what the user passed, for messages and for `Index::media`
    pub url: String,
}

/// One object of a listing.
#[derive(Debug, Clone)]
pub struct Entry {
    /// relative to the store root
    pub key: String,
    pub size: u64,
    pub mtime: u64,
    /// changes on every rewrite, even within the same second (when the backend gives one)
    pub etag: Option<String>,
}

impl Store {
    /// `s3://bucket/prefix` or a local directory (created if missing).
    pub fn open(url: &str) -> anyhow::Result<Self> {
        if let Some(rest) = url.strip_prefix("s3://") {
            let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
            let os = AmazonS3Builder::from_env()
                .with_bucket_name(bucket)
                .build()
                .with_context(|| format!("S3 store {url}"))?;
            return Ok(Self {
                os: Arc::new(os),
                root: KeyPath::from(prefix.trim_matches('/')),
                url: url.to_string(),
            });
        }
        anyhow::ensure!(
            !url.contains("://"),
            "unsupported store {url}: use s3://bucket/prefix or a local directory"
        );
        std::fs::create_dir_all(url).with_context(|| format!("creating {url}"))?;
        let dir = std::fs::canonicalize(url)?;
        Ok(Self {
            os: Arc::new(LocalFileSystem::new_with_prefix(&dir)?),
            root: KeyPath::default(),
            url: dir.display().to_string(),
        })
    }

    pub fn is_local(&self) -> bool {
        !self.url.starts_with("s3://")
    }

    fn path(&self, key: &str) -> KeyPath {
        let full = if self.root.as_ref().is_empty() {
            key.to_string()
        } else {
            format!("{}/{key}", self.root)
        };
        KeyPath::from(full.as_str())
    }

    pub async fn get(&self, key: &str) -> anyhow::Result<Vec<u8>> {
        let r = self
            .os
            .get(&self.path(key))
            .await
            .with_context(|| format!("reading {key} from {}", self.url))?;
        Ok(r.bytes().await?.to_vec())
    }

    /// `None` if the object does not exist.
    pub async fn get_opt(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>> {
        match self.os.get(&self.path(key)).await {
            Ok(r) => Ok(Some(r.bytes().await?.to_vec())),
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(e).with_context(|| format!("reading {key} from {}", self.url)),
        }
    }

    /// Single PUT: atomic for readers on S3 and on the local backend (write-then-rename).
    pub async fn put(&self, key: &str, bytes: Vec<u8>) -> anyhow::Result<()> {
        self.os
            .put(&self.path(key), PutPayload::from(bytes))
            .await
            .with_context(|| format!("writing {key} to {}", self.url))?;
        Ok(())
    }

    pub async fn delete(&self, key: &str) -> anyhow::Result<()> {
        self.os
            .delete(&self.path(key))
            .await
            .with_context(|| format!("deleting {key} from {}", self.url))?;
        Ok(())
    }

    /// Every object under `prefix` (recursive), keys relative to the store root.
    pub async fn list(&self, prefix: &str) -> anyhow::Result<Vec<Entry>> {
        let base = self.path(prefix);
        let root_len = if self.root.as_ref().is_empty() {
            0
        } else {
            self.root.as_ref().len() + 1
        };
        let metas: Vec<_> = self
            .os
            .list(Some(&base))
            .try_collect()
            .await
            .with_context(|| format!("listing {} in {}", prefix, self.url))?;
        Ok(metas
            .into_iter()
            .map(|m| Entry {
                key: m.location.as_ref()[root_len..].to_string(),
                size: m.size,
                mtime: m.last_modified.timestamp().max(0) as u64,
                etag: m.e_tag,
            })
            .collect())
    }
}

/// Reads a file given as a local path or an `s3://bucket/key` URL (labels TSV, catalogue…).
pub async fn read_any(url: &str) -> anyhow::Result<Vec<u8>> {
    if url.starts_with("s3://") {
        let (dir, key) = url.rsplit_once('/').context("s3 URL without key")?;
        return Store::open(dir)?.get(key).await;
    }
    tokio::fs::read(url)
        .await
        .with_context(|| format!("reading {url}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn local_store_roundtrip_and_listing() {
        let dir = std::env::temp_dir().join(format!("earwig-store-test-{}", std::process::id()));
        let s = Store::open(dir.to_str().unwrap()).unwrap();
        assert_eq!(s.get_opt("latest").await.unwrap(), None);
        s.put("shards/r1/0-of-2.bin", b"a".to_vec()).await.unwrap();
        s.put("shards/r1/1-of-2.bin", b"bb".to_vec()).await.unwrap();
        s.put("latest", b"snapshots/r1.bin".to_vec()).await.unwrap();
        let mut keys: Vec<_> = s
            .list("shards/r1")
            .await
            .unwrap()
            .into_iter()
            .map(|e| (e.key, e.size))
            .collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                ("shards/r1/0-of-2.bin".into(), 1),
                ("shards/r1/1-of-2.bin".into(), 2)
            ]
        );
        assert_eq!(s.get("latest").await.unwrap(), b"snapshots/r1.bin");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
