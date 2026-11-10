# zg-metis-adapter

A small, standalone Rust bridge that exposes a local OpenAI-compatible
`/v1/embeddings` endpoint and translates requests to the fixed GitHub Copilot
Metis endpoint (`https://api.github.com/embeddings`). It is intentionally narrow:
it only supports `metis-1024-I16-Binary`, and is not a general-purpose embedding
proxy or an official GitHub/Copilot project.

## Data and security

- Request text is forwarded to GitHub's fixed Metis endpoint for embedding. Do
  not submit confidential, personal, or third-party text unless you are
  authorized to do so and accept that transfer.
- Request text and credential values are never written to disk or logged.
  Successfully validated vectors are persisted in a private SQLite cache for
  seven days so failed/retried indexing runs can reuse them. Cache identity is a
  SHA-256 digest over the exact input, purpose, endpoint/protocol, model, and
  upstream credential scope; raw input and token values are not stored. Cache
  hits do not extend the expiration.
- The default database is
  `${XDG_CACHE_HOME:-~/.cache}/zg-metis-adapter/embeddings.sqlite3`; set
  `ZG_METIS_ADAPTER_CACHE_PATH` to an absolute path to override it. The adapter
  opens the cache before listening and refuses uncached calls if the cache
  cannot be opened, read, or durably committed. On Unix, its directory/database
  permissions are restricted to `0700`/`0600`. Expired rows are removed during
  cache access/startup.
- The listener defaults to `127.0.0.1:4142` and rejects non-loopback bind
  addresses. `/v1/embeddings` requires a separate local bearer token. `/healthz`
  is unauthenticated and returns only `ok`.
- The upstream Copilot OAuth token and local client token are read only from
  process environment variables. Never put token values in source, the model
  descriptor, shell history, or committed configuration. Use a protected
  secret-injection mechanism appropriate to your environment.
- The HTTP client connects directly, disables environment proxies, and refuses
  redirects. Environments that require a monitored/mandatory proxy are not
  supported by this implementation.
- Request body, input count, input bytes, response bytes, timeout, model and
  dimensions are bounded/validated. Cache misses are serialized within one
  process to coalesce duplicate requests; multiple adapter processes do not
  share an in-flight lock. There is no global request-rate limiter; keep one
  listener local and restrict access to the client token.
- The adapter refuses to start when `COPILOT_API_ENTERPRISE_URL` is set. Its
  endpoint and model are fixed in the source.

This is provided as-is. GitHub may change the upstream API or access
requirements. Verify the current service terms and your authorization before
use; the maintainers do not provide or endorse access to GitHub Copilot.

## Build and run

Requires Rust 1.98 or newer.

```sh
cargo test --locked
cargo build --release --locked
```

Set these variables in the adapter process through a secret manager or another
private mechanism:

- `ZG_METIS_COPILOT_TOKEN`: GitHub Copilot OAuth token used for the fixed
  upstream request.
- `ZG_METIS_ADAPTER_CLIENT_TOKEN`: a distinct random bearer token required by
  local embedding clients.
- `ZG_METIS_ADAPTER_LISTEN` (optional): loopback `SocketAddr`; defaults to
  `127.0.0.1:4142`.
- `ZG_METIS_ADAPTER_CACHE_PATH` (optional): absolute SQLite database path; by
  default the database is stored below `XDG_CACHE_HOME` or `~/.cache`.

After those variables are provided securely to the process, start it with:

```sh
cargo run --release --locked
```

The listener prints its address and fixed upstream URL, but does not print
secrets. Send an OpenAI-style request to
`http://127.0.0.1:4142/v1/embeddings` with `model` set to
`metis-1024-I16-Binary`, `input` as a string or string array, and `input_type`
set to `query` or `document`. Supply the local client token as a Bearer token.
Optional `dimensions` must be 1024 and `encoding_format` must be `float`.

The adapter preserves upstream status codes and `Retry-After`, and translates
validated Metis vectors into OpenAI-compatible response objects. Each complete
validated batch is durably cached before a successful response is returned. A
request sends only uncached inputs to GitHub; unchanged retry/rebuild requests
reuse the same per-input vectors for up to seven days. No content is sent until
a client makes an authenticated embedding request.

## Using with zg

The adapter can be selected by zg's generic OpenAI-compatible HTTP embedding
backend. Configure its endpoint as
`http://127.0.0.1:4142/v1/embeddings`, model as
`metis-1024-I16-Binary`, dimension as `1024`, and purpose field as
`input_type`. Configure the matching client-token environment-variable name in
zg; keep the token value private. zg workspace consent protects the request to
the local adapter, but does not itself authorize the adapter's separate request
to GitHub. Use synthetic inputs unless you have independently authorized the
upstream transfer. The adapter cache is separate from the workstation's native
Metis-provider cache and is not shared with other adapter installations.

See the [zvec-grep generic HTTP embedding configuration](https://github.com/zvec-ai/zvec-grep#runtime-configured-openai-compatible-embedding-servers)
for the descriptor format.
