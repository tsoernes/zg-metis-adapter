//! Loopback OpenAI-compatible bridge for GitHub's reviewed Metis embedding API.
//!
//! The bridge intentionally uses explicit environment variables for both its
//! local caller token and upstream GitHub Copilot token. It never logs request
//! bodies, vectors, or credential values.

use std::{env, io, net::SocketAddr, sync::Arc, time::Duration};

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
const MODEL: &str = "metis-1024-I16-Binary";
const DIMENSION: usize = 1024;
const MAX_BODY_BYTES: usize = 1_048_576;
const MAX_INPUTS: usize = 16;
const MAX_INPUT_BYTES: usize = 1_048_576;
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone)]
struct AppState {
    client_token: Arc<str>,
    upstream_token: Arc<str>,
    upstream_url: String,
    http: reqwest::Client,
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
    let upstream = state
        .http
        .post(&state.upstream_url)
        .bearer_auth(&state.upstream_token)
        .header("User-Agent", "zg-metis-adapter")
        .header("X-GitHub-Api-Version", "2025-10-01")
        .header("Copilot-Integration-Id", "code-oss")
        .json(&payload)
        .send()
        .await;
    let upstream = match upstream {
        Ok(response) => response,
        Err(_) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                "GitHub Metis embedding request failed",
            );
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
        return response;
    }
    if upstream
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
    {
        return error_response(
            StatusCode::BAD_GATEWAY,
            "GitHub Metis response was too large",
        );
    }
    let response_body = match read_bounded_body(upstream, MAX_RESPONSE_BYTES).await {
        Ok(bytes) => bytes,
        Err(()) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                "GitHub Metis response was too large",
            );
        }
    };
    let response = match serde_json::from_slice::<MetisResponse>(&response_body) {
        Ok(response) => response,
        Err(_) => {
            return error_response(StatusCode::BAD_GATEWAY, "GitHub Metis response was invalid");
        }
    };
    let expected_count = payload["inputs"]
        .as_array()
        .map_or(0, |inputs| inputs.len());
    match openai_response(response, expected_count) {
        Ok(value) => (StatusCode::OK, axum::Json(value)).into_response(),
        Err(message) => error_response(StatusCode::BAD_GATEWAY, message),
    }
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

fn metis_payload(request: OpenAiRequest) -> Result<Value, &'static str> {
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
    Ok(json!({
        "inputs": inputs,
        "input_type": request.input_type,
        "embedding_model": MODEL
    }))
}

fn openai_response(response: MetisResponse, expected_count: usize) -> Result<Value, &'static str> {
    if response.embedding_model != MODEL || response.embeddings.len() != expected_count {
        return Err("GitHub Metis response did not match the request");
    }
    let mut data = Vec::with_capacity(expected_count);
    for (index, item) in response.embeddings.into_iter().enumerate() {
        if item.embedding.len() != DIMENSION
            || item.embedding.iter().any(|value| {
                !value.is_finite() || *value > f64::from(f32::MAX) || *value < f64::from(f32::MIN)
            })
        {
            return Err("GitHub Metis returned an invalid embedding vector");
        }
        data.push(json!({
            "object": "embedding",
            "index": index,
            "embedding": item.embedding
        }));
    }
    Ok(json!({
        "object": "list",
        "data": data,
        "model": MODEL
    }))
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
    use std::net::SocketAddr;
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
            metis_payload(request).expect("Metis payload"),
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
        let response = openai_response(
            MetisResponse {
                embedding_model: MODEL.to_owned(),
                embeddings: vec![MetisEmbedding { embedding: vectors }],
            },
            1,
        )
        .expect("OpenAI response");
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
