use std::{
    env, fs,
    io::{self, ErrorKind},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use sha2::{Digest, Sha256};

const CACHE_TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);
const CACHE_SCHEMA_VERSION: &[u8] = b"zg-metis-adapter-cache-v1";
const VECTOR_DIMENSION: usize = 1024;
const CACHE_PATH_ENV: &str = "ZG_METIS_ADAPTER_CACHE_PATH";

pub(crate) struct EmbeddingCache {
    connection: Mutex<Connection>,
}

impl EmbeddingCache {
    pub(crate) fn open_default() -> io::Result<Self> {
        let path = if let Some(path) = env::var_os(CACHE_PATH_ENV) {
            PathBuf::from(path)
        } else {
            let cache_home = env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
                .ok_or_else(cache_io_error)?;
            cache_home.join("zg-metis-adapter/embeddings.sqlite3")
        };
        if !path.is_absolute() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "embedding cache path must be absolute",
            ));
        }
        Self::open_path(&path)
    }

    pub(crate) fn open_path(path: &Path) -> io::Result<Self> {
        if !path.is_absolute() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "embedding cache path must be absolute",
            ));
        }
        let parent = path.parent().ok_or_else(cache_io_error)?;
        fs::create_dir_all(parent)?;
        let parent_metadata = fs::symlink_metadata(parent)?;
        if !parent_metadata.is_dir() || parent_metadata.file_type().is_symlink() {
            return Err(cache_io_error());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
        }
        match fs::symlink_metadata(path) {
            Ok(metadata) if !metadata.is_file() || metadata.file_type().is_symlink() => {
                return Err(cache_io_error());
            }
            Ok(_) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }

        let connection = Connection::open(path).map_err(|_| cache_io_error())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        }
        Self::from_connection(connection)
    }

    #[cfg(test)]
    pub(crate) fn open_in_memory() -> io::Result<Self> {
        let connection = Connection::open_in_memory().map_err(|_| cache_io_error())?;
        Self::from_connection(connection)
    }

    fn from_connection(connection: Connection) -> io::Result<Self> {
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|_| cache_io_error())?;
        connection
            .pragma_update(None, "journal_mode", "WAL")
            .map_err(|_| cache_io_error())?;
        connection
            .pragma_update(None, "synchronous", "FULL")
            .map_err(|_| cache_io_error())?;
        connection
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key BLOB PRIMARY KEY CHECK(length(cache_key) = 32),
                    expires_at INTEGER NOT NULL,
                    vector BLOB NOT NULL
                ) WITHOUT ROWID;
                CREATE INDEX IF NOT EXISTS embeddings_expiry ON embeddings(expires_at);",
            )
            .map_err(|_| cache_io_error())?;
        connection
            .execute(
                "DELETE FROM embeddings WHERE expires_at <= ?1",
                [unix_time()?],
            )
            .map_err(|_| cache_io_error())?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    pub(crate) async fn get_many(
        cache: Arc<Self>,
        keys: Vec<[u8; 32]>,
    ) -> Result<Vec<Option<Vec<f64>>>, ()> {
        tokio::task::spawn_blocking(move || cache.get_many_sync(&keys))
            .await
            .map_err(|_| ())?
    }

    pub(crate) async fn put_many(
        cache: Arc<Self>,
        keys: Vec<[u8; 32]>,
        vectors: Vec<Vec<f64>>,
    ) -> Result<(), ()> {
        tokio::task::spawn_blocking(move || cache.put_many_sync(&keys, &vectors))
            .await
            .map_err(|_| ())?
    }

    #[cfg(test)]
    pub(crate) fn insert_raw_for_test(&self, key: [u8; 32], vector: &[u8]) -> Result<(), ()> {
        let expiry =
            unix_time().map_err(|_| ())? + i64::try_from(CACHE_TTL.as_secs()).map_err(|_| ())?;
        let connection = self.connection.lock().map_err(|_| ())?;
        connection
            .execute(
                "INSERT OR REPLACE INTO embeddings (cache_key, expires_at, vector) VALUES (?1, ?2, ?3)",
                params![key.as_slice(), expiry, vector],
            )
            .map_err(|_| ())?;
        Ok(())
    }

    fn get_many_sync(&self, keys: &[[u8; 32]]) -> Result<Vec<Option<Vec<f64>>>, ()> {
        let connection = self.connection.lock().map_err(|_| ())?;
        let now = unix_time().map_err(|_| ())?;
        connection
            .execute("DELETE FROM embeddings WHERE expires_at <= ?1", [now])
            .map_err(|_| ())?;
        let mut statement = connection
            .prepare_cached(
                "SELECT vector FROM embeddings WHERE cache_key = ?1 AND expires_at > ?2",
            )
            .map_err(|_| ())?;
        keys.iter()
            .map(|key| {
                let encoded = statement
                    .query_row(params![key.as_slice(), now], |row| row.get::<_, Vec<u8>>(0))
                    .optional()
                    .map_err(|_| ())?;
                match encoded {
                    Some(encoded) => decode_vector(&encoded).map(Some).ok_or(()),
                    None => Ok(None),
                }
            })
            .collect()
    }

    fn put_many_sync(&self, keys: &[[u8; 32]], vectors: &[Vec<f64>]) -> Result<(), ()> {
        if keys.len() != vectors.len() {
            return Err(());
        }
        if keys.is_empty() {
            return Ok(());
        }
        let expiry = unix_time()
            .map_err(|_| ())?
            .checked_add(i64::try_from(CACHE_TTL.as_secs()).map_err(|_| ())?)
            .ok_or(())?;
        if vectors.iter().any(|vector| {
            vector.len() != VECTOR_DIMENSION
                || vector
                    .iter()
                    .any(|value| !value.is_finite() || value.abs() > f64::from(f32::MAX))
        }) {
            return Err(());
        }
        let mut connection = self.connection.lock().map_err(|_| ())?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| ())?;
        {
            let mut statement = transaction
                .prepare_cached(
                    "INSERT OR REPLACE INTO embeddings (cache_key, expires_at, vector)
                     VALUES (?1, ?2, ?3)",
                )
                .map_err(|_| ())?;
            for (key, vector) in keys.iter().zip(vectors) {
                let encoded = encode_vector(vector);
                statement
                    .execute(params![key.as_slice(), expiry, encoded])
                    .map_err(|_| ())?;
            }
        }
        transaction.commit().map_err(|_| ())
    }
}

pub(crate) fn cache_key(
    endpoint: &str,
    api_version: &str,
    integration_id: &str,
    model: &str,
    purpose: &str,
    input: &str,
    credential_token: &str,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    for value in [
        CACHE_SCHEMA_VERSION,
        endpoint.as_bytes(),
        api_version.as_bytes(),
        integration_id.as_bytes(),
        model.as_bytes(),
        purpose.as_bytes(),
        input.as_bytes(),
        credential_token.as_bytes(),
    ] {
        digest.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
        digest.update(value);
    }
    digest.finalize().into()
}

fn encode_vector(vector: &[f64]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(std::mem::size_of_val(vector));
    for value in vector {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    encoded
}

fn decode_vector(encoded: &[u8]) -> Option<Vec<f64>> {
    if encoded.len() != VECTOR_DIMENSION.checked_mul(std::mem::size_of::<f64>())? {
        return None;
    }
    let (chunks, remainder) = encoded.as_chunks::<{ std::mem::size_of::<f64>() }>();
    if !remainder.is_empty() {
        return None;
    }
    let vector = chunks
        .iter()
        .map(|bytes| f64::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    vector
        .iter()
        .all(|value| value.is_finite() && value.abs() <= f64::from(f32::MAX))
        .then_some(vector)
}

fn unix_time() -> io::Result<i64> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| cache_io_error())?
        .as_secs();
    i64::try_from(seconds).map_err(|_| cache_io_error())
}

fn cache_io_error() -> io::Error {
    io::Error::other("embedding cache is unavailable")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key(purpose: &str, input: &str) -> [u8; 32] {
        cache_key(
            "https://api.github.com/embeddings",
            "2025-10-01",
            "code-oss",
            "metis-1024-I16-Binary",
            purpose,
            input,
            "synthetic upstream token",
        )
    }

    #[test]
    fn keys_bind_endpoint_model_protocol_purpose_token_scope_and_exact_input() {
        let key = test_key("query", "fixture one");
        assert_ne!(
            key,
            cache_key(
                "https://other.test/embeddings",
                "2025-10-01",
                "code-oss",
                "metis-1024-I16-Binary",
                "query",
                "fixture one",
                "synthetic upstream token"
            )
        );
        assert_ne!(
            key,
            cache_key(
                "https://api.github.com/embeddings",
                "2025-10-02",
                "code-oss",
                "metis-1024-I16-Binary",
                "query",
                "fixture one",
                "synthetic upstream token"
            )
        );
        assert_ne!(
            key,
            cache_key(
                "https://api.github.com/embeddings",
                "2025-10-01",
                "code-oss",
                "another-model",
                "query",
                "fixture one",
                "synthetic upstream token"
            )
        );
        assert_ne!(key, test_key("document", "fixture one"));
        assert_ne!(key, test_key("query", "fixture two"));
        assert_ne!(
            key,
            cache_key(
                "https://api.github.com/embeddings",
                "2025-10-01",
                "code-oss",
                "metis-1024-I16-Binary",
                "query",
                "fixture one",
                "different synthetic token"
            )
        );
        assert!(!format!("{key:?}").contains("fixture one"));
    }

    #[test]
    fn persists_validated_vectors_for_seven_days_across_reopen() {
        let directory = tempfile::tempdir().expect("temporary cache directory");
        let path = directory.path().join("private-cache/embeddings.sqlite3");
        let key = test_key("query", "synthetic persisted test");
        let vector = vec![0.251_234_567_89_f64; VECTOR_DIMENSION];
        let cache = EmbeddingCache::open_path(&path).expect("open cache");
        cache
            .put_many_sync(&[key], std::slice::from_ref(&vector))
            .expect("write cache");

        let connection = cache.connection.lock().expect("cache lock");
        let (entry_count, expires_at): (i64, i64) = connection
            .query_row(
                "SELECT count(*), max(expires_at) FROM embeddings",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("read cache retention");
        let now = unix_time().expect("clock");
        assert_eq!(entry_count, 1);
        assert!(expires_at >= now + i64::try_from(CACHE_TTL.as_secs()).expect("ttl") - 1);
        drop(connection);
        drop(cache);

        let restarted = EmbeddingCache::open_path(&path).expect("reopen cache after restart");
        assert_eq!(
            restarted.get_many_sync(&[key]).expect("read cache"),
            vec![Some(vector)]
        );
        drop(restarted);
        for entry in fs::read_dir(path.parent().expect("cache parent")).expect("cache files") {
            let bytes = fs::read(entry.expect("cache file entry").path()).expect("cache bytes");
            assert!(
                !bytes
                    .windows(b"synthetic persisted test".len())
                    .any(|window| window == b"synthetic persisted test")
            );
            assert!(
                !bytes
                    .windows(b"synthetic upstream token".len())
                    .any(|window| window == b"synthetic upstream token")
            );
        }
    }

    #[test]
    fn expired_entries_are_deleted_and_treated_as_misses() {
        let directory = tempfile::tempdir().expect("temporary cache directory");
        let path = directory.path().join("cache/embeddings.sqlite3");
        let key = test_key("document", "synthetic expired test");
        let cache = EmbeddingCache::open_path(&path).expect("open cache");
        cache
            .put_many_sync(&[key], &[vec![0.5_f64; VECTOR_DIMENSION]])
            .expect("write cache");
        {
            let connection = cache.connection.lock().expect("cache lock");
            connection
                .execute(
                    "UPDATE embeddings SET expires_at = ?1",
                    [unix_time().expect("clock") - 1],
                )
                .expect("expire fixture");
        }
        assert_eq!(
            cache.get_many_sync(&[key]).expect("expired cache miss"),
            vec![None]
        );
        let connection = cache.connection.lock().expect("cache lock");
        let count: i64 = connection
            .query_row("SELECT count(*) FROM embeddings", [], |row| row.get(0))
            .expect("count rows");
        assert_eq!(count, 0);
    }

    #[test]
    fn rejects_corrupt_or_non_finite_cached_vectors_instead_of_treating_them_as_misses() {
        let short = vec![0_u8; 8];
        assert!(decode_vector(&short).is_none());
        let mut encoded = encode_vector(&vec![0.0_f64; VECTOR_DIMENSION]);
        encoded[..8].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(decode_vector(&encoded).is_none());
    }

    #[test]
    fn invalid_batch_is_rejected_without_writing_any_vectors() {
        let directory = tempfile::tempdir().expect("temporary cache directory");
        let path = directory.path().join("cache/embeddings.sqlite3");
        let cache = EmbeddingCache::open_path(&path).expect("open cache");
        let keys = [
            test_key("query", "synthetic valid"),
            test_key("query", "synthetic invalid"),
        ];
        let vectors = [
            vec![0.25_f64; VECTOR_DIMENSION],
            vec![f64::NAN; VECTOR_DIMENSION],
        ];
        assert!(cache.put_many_sync(&keys, &vectors).is_err());
        assert_eq!(
            cache.get_many_sync(&keys).expect("cache read"),
            vec![None, None]
        );
    }

    #[cfg(unix)]
    #[test]
    fn refuses_database_symlinks() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().expect("temporary cache directory");
        let target = directory.path().join("target.sqlite3");
        fs::write(&target, b"synthetic target").expect("write target");
        let link = directory.path().join("cache-link.sqlite3");
        symlink(&target, &link).expect("create symlink");
        assert!(EmbeddingCache::open_path(&link).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn cache_directory_and_database_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().expect("temporary cache directory");
        let path = directory.path().join("private-cache/embeddings.sqlite3");
        let _cache = EmbeddingCache::open_path(&path).expect("open cache");
        assert_eq!(
            fs::metadata(path.parent().expect("parent"))
                .expect("parent metadata")
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(path)
                .expect("database metadata")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
}
