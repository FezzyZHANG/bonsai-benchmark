#!/usr/bin/env -S deno run --allow-net --allow-env
/**
 * Anthropic Messages API logging proxy.
 *
 * Required environment variables:
 *   HOST                  hostname/address to listen on
 *   PORT                  TCP port to listen on
 *   ANTHROPIC_BASE_URL    upstream base URL, e.g. https://api.anthropic.com
 *   ANTHROPIC_AUTH_TOKEN  token sent upstream as "Authorization: Bearer <token>"
 *
 * Optional environment variables:
 *   RETRY_RATE_LIMIT_422          retries for a token rate limit reported as
 *                                 HTTP 422 (default 6, 0 disables)
 *   RETRY_RATE_LIMIT_MAX_WAIT_S   cap for one retry delay (default 20 s)
 *
 * Every request and response (headers included) is written to stdout as one
 * JSON object per line. All other output goes to stderr, so stdout is pure JSONL.
 *
 * Run:
 *   HOST=127.0.0.1 PORT=8787 \
 *   ANTHROPIC_BASE_URL=https://api.anthropic.com \
 *   ANTHROPIC_AUTH_TOKEN=sk-ant-... \
 *   deno run --allow-net --allow-env anthropic-proxy.ts
 */

const REQUIRED_ENV = [
  "HOST",
  "PORT",
  "ANTHROPIC_BASE_URL",
  "ANTHROPIC_AUTH_TOKEN",
] as const;

function fail(message: string): never {
  console.error(`anthropic-proxy: ${message}`);
  Deno.exit(1);
}

const missing = REQUIRED_ENV.filter((name) => !Deno.env.get(name)?.trim());
if (missing.length > 0) {
  fail(
    `refusing to start; missing required environment variable(s): ${
      missing.join(", ")
    }`,
  );
}

const host = Deno.env.get("HOST")!.trim();

const portRaw = Deno.env.get("PORT")!.trim();
const port = Number(portRaw);
if (!Number.isInteger(port) || port < 0 || port > 65535) {
  fail(
    `PORT must be an integer between 0 and 65535 (got ${
      JSON.stringify(portRaw)
    })`,
  );
}

const baseUrlRaw = Deno.env.get("ANTHROPIC_BASE_URL")!.trim();
let baseUrl: URL;
try {
  baseUrl = new URL(baseUrlRaw);
} catch {
  fail(
    `ANTHROPIC_BASE_URL is not a valid absolute URL (got ${
      JSON.stringify(baseUrlRaw)
    })`,
  );
}
if (baseUrl.protocol !== "http:" && baseUrl.protocol !== "https:") {
  fail(`ANTHROPIC_BASE_URL must use http or https (got ${baseUrl.protocol})`);
}
if (baseUrl.search || baseUrl.hash) {
  fail("ANTHROPIC_BASE_URL must not contain a query string or fragment");
}
if (baseUrl.username || baseUrl.password) {
  fail("ANTHROPIC_BASE_URL must not contain credentials");
}

const token = Deno.env.get("ANTHROPIC_AUTH_TOKEN")!.trim();
if (/[\r\n]/.test(token)) {
  fail("ANTHROPIC_AUTH_TOKEN must not contain CR or LF characters");
}
// Claude Code's ANTHROPIC_AUTH_TOKEN is the raw value; tolerate an explicit prefix too.
const authHeaderValue = /^Bearer\s+/i.test(token) ? token : `Bearer ${token}`;
const rawToken = token.replace(/^Bearer\s+/i, "");

// Benchmark hardening. The proxy holds a real credential, so by default it
// only forwards the Messages API paths and POST requests. Adjust via env if
// the upstream/Claude Code version needs another endpoint.
const allowedPaths = new Set(
  (Deno.env.get("ALLOWED_PATHS") ??
    "/v1/messages,/v1/messages/count_tokens,/v1/models")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0),
);
const allowedMethods = new Set(
  (Deno.env.get("ALLOWED_METHODS") ?? "POST,GET")
    .split(",")
    .map((s) => s.trim().toUpperCase())
    .filter((s) => s.length > 0),
);
const maxRequestsRaw = Deno.env.get("MAX_REQUESTS")?.trim() ?? "0";
const maxRequests = Number(maxRequestsRaw);
if (!Number.isInteger(maxRequests) || maxRequests < 0) {
  fail(`MAX_REQUESTS must be a non-negative integer (got ${JSON.stringify(maxRequestsRaw)})`);
}
const logBodies = (Deno.env.get("LOG_BODIES")?.trim() ?? "1") !== "0";

// Some gateways report an upstream per-minute token limit as HTTP 422 rather
// than 429 (cc-switch: "Rate limit reached ... Please try again in N s").
// Claude Code treats 422 as a fatal request error and does not retry it, so
// the proxy re-issues the request after the delay the upstream asks for. A 422
// that is not a token rate limit passes through unchanged.
const retryRateLimit422Raw = Deno.env.get("RETRY_RATE_LIMIT_422")?.trim() ??
  "6";
const retryRateLimit422 = Number(retryRateLimit422Raw);
if (
  !Number.isInteger(retryRateLimit422) || retryRateLimit422 < 0 ||
  retryRateLimit422 > 20
) {
  fail(
    `RETRY_RATE_LIMIT_422 must be an integer between 0 and 20 (got ${
      JSON.stringify(retryRateLimit422Raw)
    })`,
  );
}
const retryRateLimitMaxWaitRaw =
  Deno.env.get("RETRY_RATE_LIMIT_MAX_WAIT_S")?.trim() ?? "20";
const retryRateLimitMaxWaitS = Number(retryRateLimitMaxWaitRaw);
if (
  !Number.isFinite(retryRateLimitMaxWaitS) || retryRateLimitMaxWaitS < 0 ||
  retryRateLimitMaxWaitS > 120
) {
  fail(
    `RETRY_RATE_LIMIT_MAX_WAIT_S must be a number between 0 and 120 (got ${
      JSON.stringify(retryRateLimitMaxWaitRaw)
    })`,
  );
}
const retryRateLimitMaxWaitMs = retryRateLimitMaxWaitS * 1000;
const authHeaderMode = (Deno.env.get("AUTH_HEADER")?.trim() ?? "authorization").toLowerCase();
if (!["authorization", "x-api-key", "both"].includes(authHeaderMode)) {
  fail(`AUTH_HEADER must be authorization, x-api-key or both (got ${JSON.stringify(authHeaderMode)})`);
}
let servedRequests = 0;

// Strip a trailing slash so path concatenation is predictable. Any path prefix
// on the base URL is preserved (e.g. https://gateway.example/anthropic).
const upstreamPrefix = baseUrl.href.replace(/\/+$/, "");

const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "proxy-connection",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

// Values of these headers are replaced in logs so credentials never leak.
// Remove entries here if you really need to see the raw values.
const SENSITIVE_HEADERS = new Set([
  "authorization",
  "x-api-key",
  "proxy-authorization",
  "cookie",
  "set-cookie",
]);

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder("utf-8", { fatal: true });

// Serialize log writes so concurrent requests cannot interleave partial lines.
let logQueue: Promise<void> = Promise.resolve();

function logRecord(record: Record<string, unknown>): void {
  // Build the complete line first, then write it in one logical operation.
  const line = textEncoder.encode(`${JSON.stringify(record)}\n`);
  logQueue = logQueue.then(async () => {
    try {
      let offset = 0;
      while (offset < line.byteLength) {
        const written = await Deno.stdout.write(line.subarray(offset));
        if (written === 0) break;
        offset += written;
      }
    } catch (error) {
      // A broken stdout should not take the proxy down.
      console.error("anthropic-proxy: failed to write log line:", error);
    }
  });
}

function headerDropSet(headers: Headers): Set<string> {
  const drop = new Set(HOP_BY_HOP_HEADERS);
  const connection = headers.get("connection");
  if (connection) {
    for (const part of connection.split(",")) {
      const name = part.trim().toLowerCase();
      if (name) drop.add(name);
    }
  }
  return drop;
}

function copyHeaders(source: Headers, extraDrop?: Iterable<string>): Headers {
  const drop = headerDropSet(source);
  for (const name of extraDrop ?? []) drop.add(name);

  const out = new Headers();
  for (const [name, value] of source) {
    if (!drop.has(name.toLowerCase())) out.append(name, value);
  }

  // Headers iteration can combine multiple Set-Cookie fields; restore them.
  const setCookies = source.getSetCookie();
  if (setCookies.length > 0) {
    out.delete("set-cookie");
    for (const cookie of setCookies) out.append("set-cookie", cookie);
  }
  return out;
}

function headersForLog(headers: Headers): Record<string, string[]> {
  const result: Record<string, string[]> = {};
  for (const [name, value] of headers) {
    const key = name.toLowerCase();
    (result[key] ??= []).push(value);
  }
  const setCookies = headers.getSetCookie();
  if (setCookies.length > 0) result["set-cookie"] = setCookies;

  for (const name of Object.keys(result)) {
    if (SENSITIVE_HEADERS.has(name)) result[name] = ["<redacted>"];
  }
  return result;
}

interface BodyFields {
  body: unknown;
  body_encoding: "json" | "utf8" | "base64" | "none";
  body_bytes: number;
}

function bodyForLog(
  bytes: Uint8Array | undefined,
  contentEncoding: string | null,
): BodyFields {
  if (!bytes || bytes.byteLength === 0) {
    return { body: null, body_encoding: "none", body_bytes: 0 };
  }

  if (!logBodies) {
    return { body: null, body_encoding: "none", body_bytes: bytes.byteLength };
  }

  const isEncoded = contentEncoding !== null &&
    contentEncoding.trim().toLowerCase() !== "identity";

  if (!isEncoded) {
    try {
      const text = textDecoder.decode(bytes);
      try {
        return {
          body: JSON.parse(text),
          body_encoding: "json",
          body_bytes: bytes.byteLength,
        };
      } catch {
        return {
          body: text,
          body_encoding: "utf8",
          body_bytes: bytes.byteLength,
        };
      }
    } catch {
      // Not valid UTF-8; fall through to base64.
    }
  }

  return {
    body: bytes.toBase64(),
    body_encoding: "base64",
    body_bytes: bytes.byteLength,
  };
}

const tolerantDecoder = new TextDecoder("utf-8");

function isTokenRateLimit(message: string): boolean {
  return /rate limit reached/i.test(message) &&
    /(tokens per min|\bTPM\b)/i.test(message);
}

/** Delay before re-issuing a rate-limited request: the upstream's suggestion
 * when present, capped, otherwise exponential backoff. */
function rateLimitDelayMs(
  message: string,
  attempt: number,
  capMs: number,
): number {
  const match = /(?:try again|retry) in\s*([\d.]+)\s*s/i.exec(message);
  const suggestedMs = match ? Number(match[1]) * 1000 : Number.NaN;
  const backoffMs = 1000 * 2 ** (attempt - 1);
  const baseMs = Number.isFinite(suggestedMs) ? suggestedMs : backoffMs;
  const jitterMs = Math.random() * 250;
  return Math.max(250, Math.min(baseMs + jitterMs, capMs));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function concatChunks(chunks: Uint8Array[]): Uint8Array {
  if (chunks.length === 1) return chunks[0];
  let total = 0;
  for (const chunk of chunks) total += chunk.byteLength;
  const result = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

type StreamEndKind = "complete" | "upstream_error" | "client_cancel";

/**
 * Passes the upstream body through unchanged while collecting it for the
 * response log. Backpressure follows the client: pull() is called only when
 * the client is ready for more data.
 */
function loggingBodyStream(
  source: ReadableStream<Uint8Array>,
  onEnd: (chunks: Uint8Array[], kind: StreamEndKind, error?: unknown) => void,
): ReadableStream<Uint8Array> {
  const reader = source.getReader();
  const chunks: Uint8Array[] = [];
  let finished = false;

  const finish = (kind: StreamEndKind, error?: unknown) => {
    if (finished) return;
    finished = true;
    onEnd(chunks, kind, error);
  };

  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          finish("complete");
          controller.close();
          return;
        }
        if (logBodies) chunks.push(value);
        controller.enqueue(value);
      } catch (error) {
        finish("upstream_error", error);
        controller.error(error);
      }
    },
    cancel(reason) {
      finish("client_cancel", reason);
      return reader.cancel(reason);
    },
  });
}

function errorBody(
  type: string,
  message: string,
): Uint8Array<ArrayBuffer> {
  return textEncoder.encode(JSON.stringify({
    type: "error",
    error: { type, message },
  }));
}

let requestCounter = 0;

function nextId(): string {
  requestCounter += 1;
  return `${requestCounter}-${crypto.randomUUID()}`;
}

function formatAddr(addr: Deno.Addr): string {
  if ("hostname" in addr) return `${addr.hostname}:${addr.port}`;
  if ("path" in addr) return addr.path;
  return `vsock:${addr.cid}:${addr.port}`;
}

async function handleRequest(
  request: Request,
  id: string,
  startedAt: number,
  remote: string,
): Promise<Response> {
  const receivedAt = new Date().toISOString();
  const incomingUrl = new URL(request.url);
  const targetUrl = new URL(
    upstreamPrefix + incomingUrl.pathname + incomingUrl.search,
  );

  // Readiness endpoint for the sandbox harness; not forwarded and not counted
  // against MAX_REQUESTS.
  if (request.method === "GET" && incomingUrl.pathname === "/__proxy_health") {
    return new Response("ok\n", {
      status: 200,
      headers: { "content-type": "text/plain" },
    });
  }

  if (
    !allowedMethods.has(request.method) ||
    !allowedPaths.has(incomingUrl.pathname)
  ) {
    const bytes = errorBody(
      "permission_error",
      `request blocked by benchmark proxy: ${request.method} ${incomingUrl.pathname}`,
    );
    const responseHeaders = new Headers({ "content-type": "application/json" });
    logRecord({
      ts: new Date().toISOString(),
      event: "response",
      id,
      remote_addr: remote,
      status: 403,
      status_text: "Forbidden",
      headers: headersForLog(responseHeaders),
      ...bodyForLog(bytes, null),
      ttfb_ms: elapsed(startedAt),
      duration_ms: elapsed(startedAt),
      stream_end: "complete",
    });
    return new Response(bytes, {
      status: 403,
      statusText: "Forbidden",
      headers: responseHeaders,
    });
  }

  if (maxRequests > 0 && servedRequests >= maxRequests) {
    const bytes = errorBody(
      "rate_limit_error",
      `benchmark proxy request budget exhausted (${maxRequests})`,
    );
    const responseHeaders = new Headers({ "content-type": "application/json" });
    logRecord({
      ts: new Date().toISOString(),
      event: "response",
      id,
      remote_addr: remote,
      status: 429,
      status_text: "Too Many Requests",
      headers: headersForLog(responseHeaders),
      ...bodyForLog(bytes, null),
      ttfb_ms: elapsed(startedAt),
      duration_ms: elapsed(startedAt),
      stream_end: "complete",
    });
    return new Response(bytes, {
      status: 429,
      statusText: "Too Many Requests",
      headers: responseHeaders,
    });
  }
  servedRequests += 1;

  // Buffer the request body once: needed for logging, and lets fetch compute
  // a correct Content-Length when forwarding.
  let requestBytes: Uint8Array<ArrayBuffer> | undefined;
  if (request.body !== null) {
    requestBytes = new Uint8Array(await request.arrayBuffer());
  }
  const fetchForbidsBody = request.method === "GET" ||
    request.method === "HEAD";

  const outboundHeaders = copyHeaders(request.headers, [
    "host",
    "content-length",
    "expect",
    "authorization",
    "x-api-key",
  ]);
  if (authHeaderMode === "x-api-key") {
    outboundHeaders.set("x-api-key", rawToken);
  } else if (authHeaderMode === "both") {
    outboundHeaders.set("authorization", authHeaderValue);
    outboundHeaders.set("x-api-key", rawToken);
  } else {
    outboundHeaders.set("authorization", authHeaderValue);
  }
  // Deno's fetch auto-decompresses responses it negotiated itself but leaves
  // Content-Encoding in place. Asking for identity keeps the body byte-exact;
  // if the upstream compresses anyway, the raw bytes and headers still match.
  outboundHeaders.set("accept-encoding", "identity");

  const requestBodyFields = bodyForLog(
    requestBytes,
    request.headers.get("content-encoding"),
  );

  logRecord({
    ts: receivedAt,
    event: "request",
    id,
    remote_addr: remote,
    method: request.method,
    url: request.url,
    upstream_url: targetUrl.href,
    headers: headersForLog(request.headers),
    upstream_headers: headersForLog(outboundHeaders),
    ...requestBodyFields,
  });

  const badGateway = (detail: string): Response => {
    const message = `upstream request failed: ${detail}`;
    const bytes = errorBody("api_error", message);
    const responseHeaders = new Headers({ "content-type": "application/json" });
    logRecord({
      ts: new Date().toISOString(),
      event: "response",
      id,
      remote_addr: remote,
      status: 502,
      status_text: "Bad Gateway",
      headers: headersForLog(responseHeaders),
      ...bodyForLog(bytes, null),
      ttfb_ms: elapsed(startedAt),
      duration_ms: elapsed(startedAt),
      upstream_error: detail,
      retries,
    });
    return new Response(bytes, {
      status: 502,
      statusText: "Bad Gateway",
      headers: responseHeaders,
    });
  };

  const requestUpstream = () =>
    fetch(targetUrl, {
      method: request.method,
      headers: outboundHeaders,
      body: fetchForbidsBody ? undefined : requestBytes,
      redirect: "manual",
    });

  let upstream: Response;
  // Set when a 422 body has been consumed for inspection. That response is
  // returned from this buffer; upstream.body is disturbed at that point.
  let bufferedBody: Uint8Array<ArrayBuffer> | null = null;
  let retries = 0;
  while (true) {
    try {
      upstream = await requestUpstream();
    } catch (error) {
      return badGateway(error instanceof Error ? error.message : String(error));
    }

    bufferedBody = null;
    if (upstream.status !== 422 || retryRateLimit422 === 0) break;

    try {
      bufferedBody = new Uint8Array(await upstream.arrayBuffer());
    } catch (error) {
      return badGateway(error instanceof Error ? error.message : String(error));
    }
    const message = tolerantDecoder.decode(bufferedBody);
    if (!isTokenRateLimit(message) || retries >= retryRateLimit422) break;

    retries += 1;
    const delayMs = rateLimitDelayMs(
      message,
      retries,
      retryRateLimitMaxWaitMs,
    );
    logRecord({
      ts: new Date().toISOString(),
      event: "retry",
      id,
      remote_addr: remote,
      attempt: retries,
      status: upstream.status,
      reason: "upstream token rate limit reported as 422",
      delay_ms: Math.round(delayMs),
    });
    await sleep(delayMs);
  }

  const responseHeaders = copyHeaders(upstream.headers);
  const ttfb = elapsed(startedAt);

  if (bufferedBody !== null) {
    // Non-rate-limit 422, or a rate-limited 422 with retries exhausted.
    responseHeaders.delete("content-length");
    logRecord({
      ts: new Date().toISOString(),
      event: "response",
      id,
      remote_addr: remote,
      status: upstream.status,
      status_text: upstream.statusText,
      headers: headersForLog(upstream.headers),
      ...bodyForLog(bufferedBody, upstream.headers.get("content-encoding")),
      ttfb_ms: ttfb,
      duration_ms: elapsed(startedAt),
      stream_end: "complete",
      retries,
    });
    return new Response(bufferedBody, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
  }

  if (upstream.body === null) {
    logRecord({
      ts: new Date().toISOString(),
      event: "response",
      id,
      remote_addr: remote,
      status: upstream.status,
      status_text: upstream.statusText,
      headers: headersForLog(upstream.headers),
      ...bodyForLog(undefined, null),
      ttfb_ms: ttfb,
      duration_ms: elapsed(startedAt),
      stream_end: "complete",
      retries,
    });
    return new Response(null, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
  }

  const loggedBody = loggingBodyStream(
    upstream.body,
    (chunks, kind, error) => {
      const bytes = concatChunks(chunks);
      const record: Record<string, unknown> = {
        ts: new Date().toISOString(),
        event: "response",
        id,
        remote_addr: remote,
        status: upstream.status,
        status_text: upstream.statusText,
        headers: headersForLog(upstream.headers),
        ...bodyForLog(bytes, upstream.headers.get("content-encoding")),
        ttfb_ms: ttfb,
        duration_ms: elapsed(startedAt),
        stream_end: kind,
        retries,
      };
      if (error !== undefined) {
        record.error = error instanceof Error ? error.message : String(error);
      }
      logRecord(record);
    },
  );

  return new Response(loggedBody, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders,
  });
}

function elapsed(startedAt: number): number {
  return Math.round((performance.now() - startedAt) * 1000) / 1000;
}

let server: Deno.HttpServer;
try {
  server = Deno.serve(
    {
      hostname: host,
      port,
      onListen: ({ hostname, port }) => {
        console.error(
          `anthropic-proxy: listening on http://${hostname}:${port} -> ${baseUrl.origin}${
            baseUrl.pathname === "/" ? "" : baseUrl.pathname
          }`,
        );
      },
    },
    async (request, info) => {
      const id = nextId();
      const startedAt = performance.now();
      const remote = formatAddr(info.remoteAddr);
      try {
        return await handleRequest(request, id, startedAt, remote);
      } catch (error) {
        const detail = error instanceof Error
          ? error.stack ?? error.message
          : String(error);
        console.error("anthropic-proxy: request handling failed:", detail);
        const bytes = errorBody("api_error", "proxy internal error");
        const responseHeaders = new Headers({
          "content-type": "application/json",
        });
        logRecord({
          ts: new Date().toISOString(),
          event: "response",
          id,
          remote_addr: remote,
          status: 500,
          status_text: "Internal Server Error",
          headers: headersForLog(responseHeaders),
          ...bodyForLog(bytes, null),
          duration_ms: elapsed(startedAt),
          stream_end: "complete",
          proxy_error: detail,
        });
        return new Response(bytes, {
          status: 500,
          statusText: "Internal Server Error",
          headers: responseHeaders,
        });
      }
    },
  );
} catch (error) {
  fail(
    `failed to start server: ${error instanceof Error ? error.message : error}`,
  );
}

await server.finished;
