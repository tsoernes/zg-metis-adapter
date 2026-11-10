//! Loopback OpenAI-compatible bridge for GitHub's reviewed Metis embedding API.
//!
//! The bridge intentionally uses explicit environment variables for both its
//! local caller token and upstream GitHub Copilot token. It never logs request
//! bodies, vectors, or credential values. Successful validated vectors are kept
//! in a private seven-day SQLite cache; raw input and token values are not stored.

mod cache;

use std::{
    collections::{HashMap, HashSet},
    env, io,
    net::SocketAddr,
    sync::Arc,
    time::Duration,
};

use cache::{EmbeddingCache, cache_key};

use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, HeaderValue, StatusCode, header::AUTHORIZATION},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};

const LISTEN_ENV: &str = "ZG_METIS_ADAPTER_LISTEN";
const CLIENT_TOKEN_ENV: &str = "ZG_METIS_ADAPTER_CLIENT_TOKEN";
const UPSTREAM_TOKEN_ENV: &str = "ZG_METIS_COPILOT_TOKEN";
const UPSTREAM_URL: &str = "https://api.github.com/embeddings";
const UPSTREAM_API_VERSION: &str = "2025-10-01";
const COPILOT_INTEGRATION_ID: &str = "code-oss";
const MODEL: &str = "metis-1024-I16-Binary";
const DIMENSION: usize = 1024;
const MAX_BODY_BYTES: usize = 1_048_576;
const MAX_INPUTS: usize = 16;
const MAX_INPUT_BYTES: usize = 1_048_576;
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

struct AppState {
    client_token: Arc<str>,
    upstream_token: Arc<str>,
    upstream_url: String,
    http: reqwest::Client,
    cache: Arc<EmbeddingCache>,
    cache_fill: tokio::sync::Mutex<()>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenAiRequest {
    model: String,
    input: InputValues,
    input_type: String,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    encoding_format: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum InputValues {
    One(String),
    Many(Vec<String>),
}

impl InputValues {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(input) => vec![input],
            Self::Many(inputs) => inputs,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
struct MetisRequestBody {
    inputs: Vec<String>,
    input_type: String,
    embedding_model: String,
}

#[derive(Debug, Deserialize)]
struct MetisResponse {
    embedding_model: String,
    embeddings: Vec<MetisEmbedding>,
}

#[derive(Debug, Deserialize)]
struct MetisEmbedding {
    embedding: Vec<f64>,
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("zg-metis-adapter: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let listen = env::var(LISTEN_ENV)
        .ok()
        .map(|value| value.parse::<SocketAddr>())
        .transpose()?
        .unwrap_or_else(|| "127.0.0.1:4142".parse().expect("static socket address"));
    if !listen.ip().is_loopback() {
        return Err(
            io::Error::other(format!("{LISTEN_ENV} must be a loopback socket address")).into(),
        );
    }
    if env::var_os("COPILOT_API_ENTERPRISE_URL").is_some() {
        return Err(io::Error::other(
            "enterprise Copilot endpoints are not supported by the Metis adapter",
        )
        .into());
    }
    let client_token = required_secret(CLIENT_TOKEN_ENV)?;
    let upstream_token = required_secret(UPSTREAM_TOKEN_ENV)?;
    let cache = Arc::new(EmbeddingCache::open_default().map_err(|_| {
        io::Error::other("could not initialize the private embedding cache; refusing to start")
    })?);
    let http = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(60))
        .build()?;
    let state = Arc::new(AppState {
        client_token: client_token.into(),
        upstream_token: upstream_token.into(),
        upstream_url: UPSTREAM_URL.to_owned(),
        http,
        cache,
        cache_fill: tokio::sync::Mutex::new(()),
    });
    let app = app(state);
    let listener = tokio::net::TcpListener::bind(listen).await?;
    println!(
        "zg-metis-adapter listening on {}; fixed upstream is {}",
        listener.local_addr()?,
        UPSTREAM_URL
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/embeddings", post(embed))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

fn required_secret(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let value =
        env::var(name).map_err(|_| format!("required environment variable {name} is unset"))?;
    if value.trim().is_empty() {
        return Err(format!("required environment variable {name} is empty").into());
    }
    Ok(value)
}

async fn health() -> &'static str {
    "ok"
}

async fn embed(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    if !authorized(&headers, &state.client_token) {
        return error_response(StatusCode::UNAUTHORIZED, "adapter authentication required");
    }
    if body.len() > MAX_BODY_BYTES {
        return error_response(StatusCode::PAYLOAD_TOO_LARGE, "request body is too large");
    }
    let request = match serde_json::from_slice::<OpenAiRequest>(&body) {
        Ok(request) => request,
        Err(_) => return error_response(StatusCode::BAD_REQUEST, "invalid embedding request"),
    };
    let payload = match metis_payload(request) {
        Ok(payload) => payload,
        Err(message) => return error_response(StatusCode::BAD_REQUEST, message),
    };
    match embed_with_cache(&state, payload).await {
        Ok(response) => (StatusCode::OK, axum::Json(response)).into_response(),
        Err(response) => *response,
    }
}

async fn embed_with_cache(
    state: &AppState,
    payload: MetisRequestBody,
) -> Result<Value, Box<Response>> {
    let keys = payload
        .inputs
        .iter()
        .map(|input| {
            cache_key(
                &state.upstream_url,
                UPSTREAM_API_VERSION,
                COPILOT_INTEGRATION_ID,
                &payload.embedding_model,
                &payload.input_type,
                input,
                &state.upstream_token,
            )
        })
        .collect::<Vec<_>>();

    let first_read = match EmbeddingCache::get_many(Arc::clone(&state.cache), keys.clone()).await {
        Ok(entries) => entries,
        Err(()) => return Err(cache_error_response()),
    };
    if first_read.iter().all(|entry| entry.is_some()) {
        return assemble_cached_vectors(&keys, first_read);
    }

    // Serialize cache misses within this single adapter process, then recheck so
    // concurrent identical calls share the first committed provider response.
    let _fill_guard = state.cache_fill.lock().await;
    let cached = match EmbeddingCache::get_many(Arc::clone(&state.cache), keys.clone()).await {
        Ok(entries) => entries,
        Err(()) => return Err(cache_error_response()),
    };
    if cached.iter().all(|entry| entry.is_some()) {
        return assemble_cached_vectors(&keys, cached);
    }

    let mut missing_seen = HashSet::<[u8; 32]>::new();
    let mut missing_keys = Vec::new();
    let mut missing_inputs = Vec::new();
    for ((key, input), cached_vector) in keys.iter().zip(&payload.inputs).zip(&cached) {
        if cached_vector.is_some() || !missing_seen.insert(*key) {
            continue;
        }
        missing_keys.push(*key);
        missing_inputs.push(input.clone());
    }

    if !missing_inputs.is_empty() {
        let missing_payload = MetisRequestBody {
            inputs: missing_inputs,
            input_type: payload.input_type,
            embedding_model: payload.embedding_model,
        };
        let vectors = fetch_metis_vectors(state, &missing_payload).await?;
        if EmbeddingCache::put_many(
            Arc::clone(&state.cache),
            missing_keys.clone(),
            vectors.clone(),
        )
        .await
        .is_err()
        {
            // Commit the validated batch before exposing it to an indexing client.
            return Err(cache_error_response());
        }
        let mut resolved = HashMap::<[u8; 32], Vec<f64>>::new();
        for (key, vector) in missing_keys.into_iter().zip(vectors) {
            resolved.insert(key, vector);
        }
        for (key, vector) in keys.iter().zip(cached) {
            if let Some(vector) = vector {
                resolved.entry(*key).or_insert(vector);
            }
        }
        return keys
            .iter()
            .map(|key| resolved.get(key).cloned().ok_or_else(cache_error_response))
            .collect::<Result<Vec<_>, _>>()
            .map(openai_response);
    }

    assemble_cached_vectors(&keys, cached)
}

fn assemble_cached_vectors(
    keys: &[[u8; 32]],
    entries: Vec<Option<Vec<f64>>>,
) -> Result<Value, Box<Response>> {
    keys.iter()
        .zip(entries)
        .map(|(key, entry)| match entry {
            Some(vector) => Ok((*key, vector)),
            None => Err(cache_error_response()),
        })
        .collect::<Result<HashMap<_, _>, _>>()
        .and_then(|resolved| {
            keys.iter()
                .map(|key| resolved.get(key).cloned().ok_or_else(cache_error_response))
                .collect::<Result<Vec<_>, _>>()
        })
        .map(openai_response)
}

fn cache_error_response() -> Box<Response> {
    Box::new(error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "embedding cache is unavailable; refusing an uncached request",
    ))
}

async fn fetch_metis_vectors(
    state: &AppState,
    payload: &MetisRequestBody,
) -> Result<Vec<Vec<f64>>, Box<Response>> {
    let expected_count = payload.inputs.len();
    let upstream = state
        .http
        .post(&state.upstream_url)
        .bearer_auth(&state.upstream_token)
        .header("User-Agent", "zg-metis-adapter")
        .header("X-GitHub-Api-Version", UPSTREAM_API_VERSION)
        .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
        .json(payload)
        .send()
        .await;
    let upstream = match upstream {
        Ok(response) => response,
        Err(_) => {
            return Err(Box::new(error_response(
                StatusCode::BAD_GATEWAY,
                "GitHub Metis embedding request failed",
            )));
        }
    };
    let status = upstream.status();
    if !status.is_success() {
        let retry_after = upstream
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let mut response = error_response(status, "GitHub Metis embedding request was rejected");
        if let Some(retry_after) = retry_after.and_then(|value| HeaderValue::from_str(&value).ok())
        {
            response.headers_mut().insert("retry-after", retry_after);
        }
        return Err(Box::new(response));
    }
    if upstream
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
    {
        return Err(Box::new(error_response(
            StatusCode::BAD_GATEWAY,
            "GitHub Metis response was too large",
        )));
    }
    let response_body = match read_bounded_body(upstream, MAX_RESPONSE_BYTES).await {
        Ok(bytes) => bytes,
        Err(()) => {
            return Err(Box::new(error_response(
                StatusCode::BAD_GATEWAY,
                "GitHub Metis response was too large",
            )));
        }
    };
    let response = match serde_json::from_slice::<MetisResponse>(&response_body) {
        Ok(response) => response,
        Err(_) => {
            return Err(Box::new(error_response(
                StatusCode::BAD_GATEWAY,
                "GitHub Metis response was invalid",
            )));
        }
    };
    metis_vectors(response, expected_count)
        .map_err(|message| Box::new(error_response(StatusCode::BAD_GATEWAY, message)))
}

async fn read_bounded_body(mut response: reqwest::Response, limit: usize) -> Result<Vec<u8>, ()> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| ())? {
        if body.len().saturating_add(chunk.len()) > limit {
            return Err(());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn authorized(headers: &HeaderMap, expected: &str) -> bool {
    let Some(token) = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
    else {
        return false;
    };
    constant_time_eq(token.as_bytes(), expected.as_bytes())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (*left ^ *right)
        })
        == 0
}

fn metis_payload(request: OpenAiRequest) -> Result<MetisRequestBody, &'static str> {
    if request.model != MODEL {
        return Err("unsupported embedding model");
    }
    if request
        .dimensions
        .is_some_and(|dimension| dimension != DIMENSION)
    {
        return Err("Metis supports 1024 dimensions");
    }
    if request
        .encoding_format
        .as_deref()
        .is_some_and(|format| format != "float")
    {
        return Err("Metis adapter supports only float encoding");
    }
    if request.input_type != "query" && request.input_type != "document" {
        return Err("input_type must be query or document");
    }
    let inputs = request.input.into_vec();
    if inputs.is_empty() || inputs.len() > MAX_INPUTS {
        return Err("input batch size is outside supported bounds");
    }
    let input_bytes = inputs.iter().map(String::len).sum::<usize>();
    if input_bytes > MAX_INPUT_BYTES {
        return Err("input batch exceeds the byte limit");
    }
    Ok(MetisRequestBody {
        inputs,
        input_type: request.input_type,
        embedding_model: MODEL.to_owned(),
    })
}

fn metis_vectors(
    response: MetisResponse,
    expected_count: usize,
) -> Result<Vec<Vec<f64>>, &'static str> {
    if response.embedding_model != MODEL || response.embeddings.len() != expected_count {
        return Err("GitHub Metis response did not match the request");
    }
    let mut vectors = Vec::with_capacity(expected_count);
    for item in response.embeddings {
        if item.embedding.len() != DIMENSION
            || item.embedding.iter().any(|value| {
                !value.is_finite() || *value > f64::from(f32::MAX) || *value < f64::from(f32::MIN)
            })
        {
            return Err("GitHub Metis returned an invalid embedding vector");
        }
        vectors.push(item.embedding);
    }
    Ok(vectors)
}

fn openai_response(vectors: Vec<Vec<f64>>) -> Value {
    json!({
        "object": "list",
        "data": vectors
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| json!({
                "object": "embedding",
                "index": index,
                "embedding": embedding
            }))
            .collect::<Vec<_>>(),
        "model": MODEL
    })
}

fn error_response(status: StatusCode, message: &'static str) -> Response {
    (
        status,
        axum::Json(json!({
            "error": {
                "message": message,
                "type": "metis_adapter_error"
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, routing::post};
    use serde_json::Value;
    use std::{
        net::SocketAddr,
        sync::{
            Mutex as StdMutex,
            atomic::{AtomicUsize, Ordering},
        },
    };
    use tokio::task::JoinHandle;

    #[test]
    fn maps_openai_request_to_metis_schema_without_leaking_input_metadata() {
        let request: OpenAiRequest = serde_json::from_value(json!({
            "model": MODEL,
            "input": ["synthetic adapter test"],
            "input_type": "query",
            "dimensions": DIMENSION,
            "encoding_format": "float"
        }))
        .expect("OpenAI request");
        assert_eq!(
            serde_json::to_value(metis_payload(request).expect("Metis payload")).expect("JSON"),
            json!({
                "inputs": ["synthetic adapter test"],
                "input_type": "query",
                "embedding_model": MODEL
            })
        );
    }

    #[test]
    fn maps_validated_metis_vectors_to_openai_ordered_response() {
        let vectors = (0..DIMENSION)
            .map(|index| (index as f64) / 1024.0)
            .collect();
        let vectors = metis_vectors(
            MetisResponse {
                embedding_model: MODEL.to_owned(),
                embeddings: vec![MetisEmbedding { embedding: vectors }],
            },
            1,
        )
        .expect("validated vectors");
        let response = openai_response(vectors);
        assert_eq!(response["data"][0]["index"], 0);
        assert_eq!(
            response["data"][0]["embedding"]
                .as_array()
                .expect("embedding vector")
                .len(),
            DIMENSION
        );
    }

    #[test]
    fn local_adapter_requires_its_separate_bearer_token() {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_static("Bearer synthetic-client-token"),
        );
        assert!(authorized(&headers, "synthetic-client-token"));
        assert!(!authorized(&headers, "different-token"));
        headers.remove(AUTHORIZATION);
        assert!(!authorized(&headers, "synthetic-client-token"));
    }

    #[test]
    fn rejects_non_loopback_listener_addresses() {
        assert!(
            !"0.0.0.0:4142"
                .parse::<SocketAddr>()
                .expect("socket")
                .ip()
                .is_loopback()
        );
        assert!(
            "127.0.0.1:4142"
                .parse::<SocketAddr>()
                .expect("socket")
                .ip()
                .is_loopback()
        );
    }

    async fn mock_metis(Json(request): Json<Value>) -> Json<Value> {
        assert_eq!(request["input_type"], "query");
        assert_eq!(request["inputs"], json!(["synthetic route test"]));
        Json(json!({
            "embedding_model": MODEL,
            "embeddings": [{"embedding": vec![0.25_f64; DIMENSION]}]
        }))
    }

    async fn mock_rate_limited() -> Response {
        let mut response = (StatusCode::TOO_MANY_REQUESTS, "synthetic rate limit").into_response();
        response
            .headers_mut()
            .insert("retry-after", HeaderValue::from_static("2"));
        response
    }

    #[derive(Clone)]
    struct CountedUpstreamState {
        calls: Arc<AtomicUsize>,
        inputs: Arc<StdMutex<Vec<(String, String)>>>,
    }

    async fn mock_metis_counted(
        State(state): State<CountedUpstreamState>,
        Json(request): Json<Value>,
    ) -> Json<Value> {
        state.calls.fetch_add(1, Ordering::SeqCst);
        let purpose = request["input_type"]
            .as_str()
            .expect("input type")
            .to_owned();
        let inputs = request["inputs"].as_array().expect("input array");
        let mut embeddings = Vec::new();
        for input in inputs {
            let input = input.as_str().expect("input text").to_owned();
            state
                .inputs
                .lock()
                .expect("input log")
                .push((purpose.clone(), input.clone()));
            let value = if input == "synthetic alpha" {
                0.25_f64
            } else {
                0.75_f64
            };
            embeddings.push(json!({"embedding": vec![value; DIMENSION]}));
        }
        Json(json!({"embedding_model": MODEL, "embeddings": embeddings}))
    }

    fn test_state(upstream_url: String, cache: Arc<EmbeddingCache>) -> Arc<AppState> {
        Arc::new(AppState {
            client_token: Arc::from("synthetic-client-token"),
            upstream_token: Arc::from("synthetic-upstream-token"),
            upstream_url,
            http: reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("test HTTP client"),
            cache,
            cache_fill: tokio::sync::Mutex::new(()),
        })
    }

    async fn send_test_body(
        client: &reqwest::Client,
        address: SocketAddr,
        body: &Value,
    ) -> reqwest::Response {
        client
            .post(format!("http://{address}/v1/embeddings"))
            .bearer_auth("synthetic-client-token")
            .json(body)
            .send()
            .await
            .expect("synthetic adapter request")
    }

    async fn serve(app: Router) -> (SocketAddr, JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test listener");
        let address = listener.local_addr().expect("test listener address");
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("test server");
        });
        (address, task)
    }

    #[tokio::test]
    async fn openai_embeddings_route_bridges_only_synthetic_input() {
        let upstream = Router::new().route("/embeddings", post(mock_metis));
        let (upstream_address, upstream_task) = serve(upstream).await;
        let http = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("test HTTP client");
        let state = Arc::new(AppState {
            client_token: Arc::from("synthetic-client-token"),
            upstream_token: Arc::from("synthetic-upstream-token"),
            upstream_url: format!("http://{upstream_address}/embeddings"),
            http,
            cache: Arc::new(EmbeddingCache::open_in_memory().expect("test cache")),
            cache_fill: tokio::sync::Mutex::new(()),
        });
        let (adapter_address, adapter_task) = serve(app(state)).await;
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("test caller");
        let unauthorized = client
            .post(format!("http://{adapter_address}/v1/embeddings"))
            .json(&json!({
                "model": MODEL,
                "input": "synthetic route test",
                "input_type": "query"
            }))
            .send()
            .await
            .expect("unauthorized adapter request");
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        let response = client
            .post(format!("http://{adapter_address}/v1/embeddings"))
            .bearer_auth("synthetic-client-token")
            .json(&json!({
                "model": MODEL,
                "input": "synthetic route test",
                "input_type": "query"
            }))
            .send()
            .await
            .expect("adapter request");
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value = response.json().await.expect("OpenAI response JSON");
        assert_eq!(body["model"], MODEL);
        assert_eq!(body["data"][0]["index"], 0);
        assert_eq!(
            body["data"][0]["embedding"]
                .as_array()
                .expect("embedding vector")
                .len(),
            DIMENSION
        );
        adapter_task.abort();
        upstream_task.abort();
    }

    #[tokio::test]
    async fn persisted_cache_coalesces_concurrent_duplicates_and_preserves_order() {
        let directory = tempfile::tempdir().expect("temporary cache directory");
        let cache_path = directory.path().join("private-cache/embeddings.sqlite3");
        let upstream_state = CountedUpstreamState {
            calls: Arc::new(AtomicUsize::new(0)),
            inputs: Arc::new(StdMutex::new(Vec::new())),
        };
        let upstream = Router::new()
            .route("/embeddings", post(mock_metis_counted))
            .with_state(upstream_state.clone());
        let (upstream_address, upstream_task) = serve(upstream).await;
        let upstream_url = format!("http://{upstream_address}/embeddings");
        let cache =
            Arc::new(EmbeddingCache::open_path(&cache_path).expect("open persistent cache"));
        let (adapter_address, adapter_task) =
            serve(app(test_state(upstream_url.clone(), cache))).await;
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("test client");
        let body = json!({
            "model": MODEL,
            "input": ["synthetic alpha", "synthetic beta", "synthetic alpha"],
            "input_type": "query"
        });

        let first = send_test_body(&client, adapter_address, &body);
        let second = send_test_body(&client, adapter_address, &body);
        let (first, second) = tokio::join!(first, second);
        let first: Value = first.json().await.expect("first JSON");
        let second: Value = second.json().await.expect("second JSON");
        for response in [&first, &second] {
            assert_eq!(response["data"][0]["index"], 0);
            assert_eq!(response["data"][1]["index"], 1);
            assert_eq!(response["data"][2]["index"], 2);
            assert_eq!(response["data"][0]["embedding"][0], 0.25);
            assert_eq!(response["data"][1]["embedding"][0], 0.75);
            assert_eq!(
                response["data"][2]["embedding"],
                response["data"][0]["embedding"]
            );
        }
        assert_eq!(upstream_state.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            *upstream_state.inputs.lock().expect("input log"),
            vec![
                ("query".to_owned(), "synthetic alpha".to_owned()),
                ("query".to_owned(), "synthetic beta".to_owned())
            ]
        );
        adapter_task.abort();
        let _ = adapter_task.await;

        // A newly constructed cache and server emulate an adapter process restart.
        let restarted_cache =
            Arc::new(EmbeddingCache::open_path(&cache_path).expect("reopen cache"));
        let (restarted_address, restarted_task) =
            serve(app(test_state(upstream_url.clone(), restarted_cache))).await;
        let restarted_body: Value = send_test_body(&client, restarted_address, &body)
            .await
            .json()
            .await
            .expect("restarted cache response");
        assert_eq!(restarted_body, first);
        assert_eq!(upstream_state.calls.load(Ordering::SeqCst), 1);

        // Mixed batches fetch only the missing input and retain original ordering.
        let mixed_body = json!({
            "model": MODEL,
            "input": ["synthetic alpha", "synthetic gamma"],
            "input_type": "query"
        });
        let mixed: Value = send_test_body(&client, restarted_address, &mixed_body)
            .await
            .json()
            .await
            .expect("mixed cache response");
        assert_eq!(mixed["data"][0]["embedding"][0], 0.25);
        assert_eq!(mixed["data"][1]["embedding"][0], 0.75);
        assert_eq!(upstream_state.calls.load(Ordering::SeqCst), 2);
        let last_input = upstream_state
            .inputs
            .lock()
            .expect("input log")
            .last()
            .cloned();
        assert_eq!(
            last_input,
            Some(("query".to_owned(), "synthetic gamma".to_owned()))
        );

        // Query/document purpose is part of the key, so a document embedding is distinct.
        let document_body = json!({
            "model": MODEL,
            "input": "synthetic alpha",
            "input_type": "document"
        });
        let document_response = send_test_body(&client, restarted_address, &document_body).await;
        assert_eq!(document_response.status(), StatusCode::OK);
        assert_eq!(upstream_state.calls.load(Ordering::SeqCst), 3);
        restarted_task.abort();
        let _ = restarted_task.await;
        upstream_task.abort();
    }

    #[tokio::test]
    async fn corrupt_cache_entry_fails_closed_without_calling_upstream() {
        let upstream_state = CountedUpstreamState {
            calls: Arc::new(AtomicUsize::new(0)),
            inputs: Arc::new(StdMutex::new(Vec::new())),
        };
        let upstream = Router::new()
            .route("/embeddings", post(mock_metis_counted))
            .with_state(upstream_state.clone());
        let (upstream_address, upstream_task) = serve(upstream).await;
        let upstream_url = format!("http://{upstream_address}/embeddings");
        let cache = Arc::new(EmbeddingCache::open_in_memory().expect("test cache"));
        let key = cache_key(
            &upstream_url,
            UPSTREAM_API_VERSION,
            COPILOT_INTEGRATION_ID,
            MODEL,
            "query",
            "synthetic corrupt cache input",
            "synthetic-upstream-token",
        );
        cache
            .insert_raw_for_test(key, &[0; 8])
            .expect("insert malformed cache fixture");
        let (adapter_address, adapter_task) = serve(app(test_state(upstream_url, cache))).await;
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("test client");
        let response = send_test_body(
            &client,
            adapter_address,
            &json!({
                "model": MODEL,
                "input": "synthetic corrupt cache input",
                "input_type": "query"
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(upstream_state.calls.load(Ordering::SeqCst), 0);
        adapter_task.abort();
        upstream_task.abort();
    }

    #[tokio::test]
    async fn openai_route_forwards_upstream_retry_after_without_buffering_body() {
        let upstream = Router::new().route("/embeddings", post(mock_rate_limited));
        let (upstream_address, upstream_task) = serve(upstream).await;
        let http = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("test HTTP client");
        let state = Arc::new(AppState {
            client_token: Arc::from("synthetic-client-token"),
            upstream_token: Arc::from("synthetic-upstream-token"),
            upstream_url: format!("http://{upstream_address}/embeddings"),
            http,
            cache: Arc::new(EmbeddingCache::open_in_memory().expect("test cache")),
            cache_fill: tokio::sync::Mutex::new(()),
        });
        let (adapter_address, adapter_task) = serve(app(state)).await;
        let response = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("test caller")
            .post(format!("http://{adapter_address}/v1/embeddings"))
            .bearer_auth("synthetic-client-token")
            .json(&json!({
                "model": MODEL,
                "input": "synthetic retry test",
                "input_type": "query"
            }))
            .send()
            .await
            .expect("adapter request");
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response.headers().get("retry-after").expect("retry header"),
            "2"
        );
        adapter_task.abort();
        upstream_task.abort();
    }
}
