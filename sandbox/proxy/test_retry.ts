// Exercises the proxy's retry of a token rate limit reported as HTTP 422.
//
// Runs with the proxy source and this file inside the bonsai-proxy image:
//   podman run --rm \
//     -v "$PWD/sandbox/proxy:/proxy:ro" \
//     bonsai-proxy:1.0 deno run --allow-all /proxy/test_retry.ts
//
// A mock upstream answers the first two requests with the cc-switch-style 422
// ("Rate limit reached ... tokens per min (TPM) ... try again in Ns"), then
// succeeds. The proxy must hide that from the client. A 422 that is not a
// token rate limit must pass through unchanged, and retries must be bounded.

const mockPort = 19001;
const proxyPort = 19002;
const retries = 3;

type Mode = "limit-then-ok" | "limit-forever" | "invalid";
let mode: Mode = "limit-then-ok";
let hits = 0;

const rateLimitMessage =
  "Rate limit reached for gpt-6.1-sol in organization org-x on tokens per min " +
  "(TPM): Limit 1000000, Used 978374, Requested 137139. " +
  "Please try again in 0.05s.";

function errorResponse(message: string): Response {
  return new Response(
    JSON.stringify({
      type: "error",
      error: { type: "invalid_request_error", message },
    }),
    { status: 422, headers: { "content-type": "application/json" } },
  );
}

const upstream = Deno.serve({ hostname: "127.0.0.1", port: mockPort }, () => {
  hits += 1;
  if (mode === "invalid") return errorResponse("bad tool schema");
  if (mode === "limit-forever" || hits <= 2) {
    return errorResponse(rateLimitMessage);
  }
  return new Response("data: {}\n\n", {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
});

// Start through the least-privilege launcher so the permission list in
// anthropic-proxy.sh is exercised too (a missing --allow-env entry is a
// startup failure in the sandbox).
const proxy = new Deno.Command("/bin/bash", {
  args: ["/proxy/anthropic-proxy.sh"],
  env: {
    HOST: "127.0.0.1",
    PORT: String(proxyPort),
    ANTHROPIC_BASE_URL: `http://127.0.0.1:${mockPort}`,
    ANTHROPIC_AUTH_TOKEN: "test-token",
    LOG_BODIES: "0",
    RETRY_RATE_LIMIT_422: String(retries),
    RETRY_RATE_LIMIT_MAX_WAIT_S: "2",
  },
  stdout: "piped",
  stderr: "inherit",
}).spawn();

for (let i = 0; i < 100; i += 1) {
  try {
    const ready = await fetch(`http://127.0.0.1:${proxyPort}/__proxy_health`);
    if (ready.ok) break;
  } catch {
    // not listening yet
  }
  await new Promise((resolve) => setTimeout(resolve, 100));
}

const post = () =>
  fetch(`http://127.0.0.1:${proxyPort}/v1/messages`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "gpt-6.1-sol", messages: [] }),
  });

let failed = false;
const check = (name: string, ok: boolean, detail: string) => {
  console.log(`${ok ? "PASS" : "FAIL"} ${name}: ${detail}`);
  if (!ok) failed = true;
};

mode = "limit-then-ok";
hits = 0;
const started = Date.now();
const ok = await post();
const okBody = await ok.text();
check(
  "retry-then-success",
  ok.status === 200 && hits === 3,
  `status=${ok.status} upstream_hits=${hits} waited_ms=${
    Date.now() - started
  } body=${okBody.slice(0, 32)}`,
);

mode = "invalid";
hits = 0;
const bad = await post();
const badBody = await bad.text();
check(
  "pass-through-422",
  bad.status === 422 && hits === 1 && badBody.includes("bad tool schema"),
  `status=${bad.status} upstream_hits=${hits} body=${badBody.slice(0, 64)}`,
);

mode = "limit-forever";
hits = 0;
const limited = await post();
await limited.body?.cancel();
check(
  "retry-bounded",
  limited.status === 422 && hits === retries + 1,
  `status=${limited.status} upstream_hits=${hits} (expected ${
    retries + 1
  })`,
);

proxy.kill("SIGTERM");
const { stdout } = await proxy.output();
const retryEvents = new TextDecoder().decode(stdout).split("\n")
  .filter((line) => line.includes('"event":"retry"')).length;
check(
  "retry-events-logged",
  retryEvents === 2 + retries,
  `events=${retryEvents} (expected ${2 + retries})`,
);

upstream.shutdown();
Deno.exit(failed ? 1 : 0);
