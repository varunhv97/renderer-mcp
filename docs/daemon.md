# Persistent named scenes

Start a foreground local daemon in one terminal:

```sh
cargo run -p renderer-cli -- daemon serve --endpoint 127.0.0.1:9472
```

In another terminal, create, patch, and render a scene through the same
loopback-only endpoint. Patch files contain typed `ScenePatchV1` operations and
may include an `expected_revision` to avoid overwriting a newer scene revision.

```sh
cargo run -p renderer-cli -- scene --endpoint 127.0.0.1:9472 create demo --input examples/basic.scene.json
cargo run -p renderer-cli -- scene --endpoint 127.0.0.1:9472 get demo
cargo run -p renderer-cli -- scene --endpoint 127.0.0.1:9472 render demo --output .renderer/output/demo.png
cargo run -p renderer-cli -- scene --endpoint 127.0.0.1:9472 destroy demo
```

The daemon only ever binds a loopback address, caps requests at 1 MiB, and
rejects unknown fields/operations. Everything under `.renderer/` is generated
local output and is not committed.

## Per-session performance metrics

Each `daemon serve` run is one session. A background thread appends one JSON
line per request to `.renderer/metrics/<session-id>.jsonl` (method, scene ID,
duration, and outcome) so local performance testing doesn't need extra
instrumentation:

```sh
cat .renderer/metrics/*.jsonl | jq -c '{method, duration_ms, success}'
```

Recording only enqueues onto a channel from the request-handling path, so it
adds no latency there; if the metrics directory can't be created, the daemon
logs a warning and keeps serving without it.
