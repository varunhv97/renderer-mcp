// -- cmux native file preview --------------------------------------------
//
// cmux (a native macOS terminal for running coding agents, built on
// Ghostty) exposes a JSON-RPC-over-Unix-socket control API at the path
// named by `CMUX_SOCKET_PATH`. Its `file.open` method opens a local file in
// cmux's own native file-preview panel -- a UI surface entirely separate
// from the terminal grid, so unlike every protocol above it can never
// collide with an agent's own actively-redrawn TUI (the motivating
// problem: a detached `show` subprocess's raw pty write has no way to know
// where a chat transcript ends and an input box begins). This is tried
// first, ahead of all Kitty/iTerm2 detection, and used instead of it
// entirely whenever cmux is reachable. `UnixStream` is POSIX-only, so this
// whole integration is unix-only; on other platforms (this workspace's CI
// includes a Windows target) it compiles to a stub that always reports
// cmux as unavailable.
//
// The env-var/working-directory-reading entry points (`try_show`,
// `try_clear`) are thin wrappers around `_at`-suffixed functions that take
// the socket path and state-file path as explicit arguments -- mirroring
// `detect_protocol_auto`'s injected-`get` pattern above, and for the same
// reason: `std::env::set_var`/`set_current_dir` are process-global and
// would race across this crate's parallel test threads, so the tests
// exercise the `_at` functions directly instead.

use crate::error::TerminalError;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::os::unix::net::UnixStream;

/// Env var cmux sets to the path of its control socket.
const SOCKET_ENV: &str = "CMUX_SOCKET_PATH";
/// Read/write timeout applied to the control-socket connection, so a
/// hung or misbehaving socket can't make `show` hang forever.
const SOCKET_TIMEOUT: Duration = Duration::from_secs(3);
/// Where the most recently opened preview surface's id is persisted,
/// so a later clear can close it. Relative to the working directory,
/// following the same `.renderer/`-prefixed local generated-state
/// convention as the daemon's `.renderer/metrics` directory (see
/// `DEFAULT_METRICS_DIR` in `crates/daemon/src/lib.rs`).
const STATE_PATH: &str = ".renderer/cmux-preview-surface.json";

#[derive(Debug, Serialize)]
struct Request<'a> {
    id: &'a str,
    method: &'a str,
    params: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct Response {
    ok: bool,
    #[serde(default)]
    result: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<RpcError>,
}

#[derive(Debug, Deserialize)]
struct RpcError {
    message: String,
    #[serde(default)]
    code: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct State {
    surface_id: String,
}

/// The result of attempting a cmux `clear`, distinguishing "cmux isn't
/// available at all" (the caller should fall back to the
/// terminal-protocol clear path) from the two outcomes once cmux *is*
/// reachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CmuxClearOutcome {
    Unavailable,
    Cleared,
    NoSurface,
}

fn socket_path() -> Option<PathBuf> {
    std::env::var(SOCKET_ENV)
        .ok()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn state_path() -> PathBuf {
    PathBuf::from(STATE_PATH)
}

/// Connects to cmux's control socket, applying `SOCKET_TIMEOUT` in both
/// directions. Any failure at all (no such socket, connection refused,
/// a timeout while configuring the connection, ...) is treated as
/// "cmux isn't available" rather than an error -- per the spec, callers
/// fall back to the terminal-protocol path instead of failing `show`
/// outright.
fn connect(path: &Path) -> Option<UnixStream> {
    let stream = UnixStream::connect(path).ok()?;
    stream.set_read_timeout(Some(SOCKET_TIMEOUT)).ok()?;
    stream.set_write_timeout(Some(SOCKET_TIMEOUT)).ok()?;
    Some(stream)
}

/// The outcome of one request/response round-trip, per [`send_request`].
/// `Response` is a structurally valid JSON-RPC reply from a genuine
/// cmux peer -- whether `ok: true` or `ok: false`; callers decide how
/// to handle that. `Unavailable` covers every transport-level problem
/// (write/read failure, no response, non-JSON response) and is grouped
/// with a failed `connect` rather than treated as a hard error: a
/// socket that *accepts a connection* but doesn't speak cmux's
/// protocol correctly for this caller (for example, `CMUX_SOCKET_PATH`
/// leaked via process-environment inheritance into a shell outside the
/// cmux instance that actually owns that socket) isn't meaningfully
/// different from "cmux isn't available here" -- the caller should
/// fall back to the terminal-protocol path either way, the same as it
/// would if `connect` itself had failed.
enum RequestOutcome {
    Unavailable,
    Response(Response),
}

/// One request/response round-trip: writes a single newline-terminated
/// JSON request and reads a single newline-terminated JSON response,
/// per cmux's documented framing. See [`RequestOutcome`] for how
/// failures after a successful `connect` are classified.
fn send_request(
    stream: &mut UnixStream,
    id: &str,
    method: &str,
    params: serde_json::Value,
) -> Result<RequestOutcome, TerminalError> {
    let cmux_error = |message: String| TerminalError::Cmux { message };

    // Encoding our own well-typed params can't fail in practice, so a
    // failure here points at a local bug rather than an unusable peer
    // -- unlike everything below, it stays a hard error.
    let mut payload = serde_json::to_vec(&Request { id, method, params })
        .map_err(|source| cmux_error(format!("could not encode request: {source}")))?;
    payload.push(b'\n');
    if stream.write_all(&payload).is_err() {
        return Ok(RequestOutcome::Unavailable);
    }

    let mut line = String::new();
    if BufReader::new(&mut *stream).read_line(&mut line).is_err() {
        return Ok(RequestOutcome::Unavailable);
    }
    if line.trim().is_empty() {
        return Ok(RequestOutcome::Unavailable);
    }

    match serde_json::from_str(&line) {
        Ok(response) => Ok(RequestOutcome::Response(response)),
        Err(_) => Ok(RequestOutcome::Unavailable),
    }
}

/// [`send_request`], but an `ok: false` response is itself turned into
/// a hard `TerminalError::Cmux` -- the behavior every caller except
/// `try_clear_at` wants. `Ok(None)` propagates a `RequestOutcome::Unavailable`
/// for the caller to fall back on, same as a failed `connect`.
fn call(
    stream: &mut UnixStream,
    id: &str,
    method: &str,
    params: serde_json::Value,
) -> Result<Option<serde_json::Value>, TerminalError> {
    let cmux_error = |message: String| TerminalError::Cmux { message };
    let response = match send_request(stream, id, method, params)? {
        RequestOutcome::Unavailable => return Ok(None),
        RequestOutcome::Response(response) => response,
    };
    if !response.ok {
        let detail = response
            .error
            .map(|error| match error.code {
                Some(code) => format!("{} ({code})", error.message),
                None => error.message,
            })
            .unwrap_or_else(|| "unknown error".to_string());
        return Err(cmux_error(format!("rejected `{method}`: {detail}")));
    }
    let result = response
        .result
        .ok_or_else(|| cmux_error(format!("no result for `{method}`")))?;
    Ok(Some(result))
}

fn load_state(state_file: &Path) -> Option<State> {
    let contents = fs::read_to_string(state_file).ok()?;
    serde_json::from_str(&contents).ok()
}

/// Best-effort: an unwritable working directory shouldn't fail an
/// otherwise-successful `show`, it just means a later clear won't have
/// a recorded surface to close.
fn save_state(state_file: &Path, surface_id: &str) {
    if let Some(parent) = state_file.parent()
        && fs::create_dir_all(parent).is_err()
    {
        return;
    }
    if let Ok(json) = serde_json::to_string(&State {
        surface_id: surface_id.to_string(),
    }) {
        let _ = fs::write(state_file, json);
    }
}

fn clear_state(state_file: &Path) {
    let _ = fs::remove_file(state_file);
}

/// `run`'s display path via cmux. `Ok(None)` means cmux isn't usable
/// here -- either `CMUX_SOCKET_PATH` is unset, its socket didn't accept
/// a connection, or (see `RequestOutcome`) it accepted a connection but
/// didn't answer like a real cmux peer -- and the caller should fall
/// back to terminal-protocol detection unchanged. Once we have a
/// structurally valid response, a bad path or a rejected RPC is
/// surfaced as a hard error instead.
pub(crate) fn try_show(path: &Path) -> Result<Option<&'static str>, TerminalError> {
    let Some(socket) = socket_path() else {
        return Ok(None);
    };
    try_show_at(&socket, &state_path(), path)
}

/// The testable core of `try_show`, with the socket and state-file
/// paths taken as explicit arguments instead of read from the
/// environment/working directory.
fn try_show_at(
    socket: &Path,
    state_file: &Path,
    path: &Path,
) -> Result<Option<&'static str>, TerminalError> {
    let Some(mut stream) = connect(socket) else {
        return Ok(None);
    };
    let absolute = fs::canonicalize(path).map_err(|source| TerminalError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let Some(result) = call(
        &mut stream,
        "renderer-show",
        "file.open",
        serde_json::json!({ "path": absolute.to_string_lossy() }),
    )?
    else {
        return Ok(None);
    };
    if let Some(surface_id) = result.get("surface_id").and_then(|value| value.as_str()) {
        save_state(state_file, surface_id);
    }
    Ok(Some("cmux"))
}

/// `run`'s clear path via cmux: closes the most recently opened
/// preview surface, if one was recorded, via `surface.close`.
pub(crate) fn try_clear() -> Result<CmuxClearOutcome, TerminalError> {
    let Some(socket) = socket_path() else {
        return Ok(CmuxClearOutcome::Unavailable);
    };
    try_clear_at(&socket, &state_path())
}

/// The testable core of `try_clear`, with the socket and state-file
/// paths taken as explicit arguments.
fn try_clear_at(socket: &Path, state_file: &Path) -> Result<CmuxClearOutcome, TerminalError> {
    let Some(mut stream) = connect(socket) else {
        return Ok(CmuxClearOutcome::Unavailable);
    };
    let Some(state) = load_state(state_file) else {
        return Ok(CmuxClearOutcome::NoSurface);
    };
    let response = match send_request(
        &mut stream,
        "renderer-clear",
        "surface.close",
        serde_json::json!({ "surface_id": state.surface_id }),
    )? {
        RequestOutcome::Unavailable => return Ok(CmuxClearOutcome::Unavailable),
        RequestOutcome::Response(response) => response,
    };
    // The recorded surface (or the workspace/window it lived in) may
    // already be gone -- the user closed the tab, or a stale state file
    // survived from an earlier, now-defunct cmux session. `--clear` is
    // meant to be idempotent ("make sure nothing is showing"), so
    // treat cmux telling us the target is already gone as success
    // rather than an error, the same as `CmuxClearOutcome::NoSurface`.
    // Any other rejection (a real transport/application error) still
    // propagates as a hard failure.
    if !response.ok {
        let code = response
            .error
            .as_ref()
            .and_then(|error| error.code.as_deref());
        if code == Some("not_found") {
            clear_state(state_file);
            return Ok(CmuxClearOutcome::NoSurface);
        }
        let detail = response
            .error
            .map(|error| match error.code {
                Some(code) => format!("{} ({code})", error.message),
                None => error.message,
            })
            .unwrap_or_else(|| "unknown error".to_string());
        return Err(TerminalError::Cmux {
            message: format!("rejected `surface.close`: {detail}"),
        });
    }
    clear_state(state_file);
    Ok(CmuxClearOutcome::Cleared)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;

    /// Spawns a fake cmux control-socket server bound to a tempdir
    /// path: accepts exactly one connection, reads one
    /// newline-terminated request, and writes back `response` (with a
    /// trailing newline added if it doesn't already end in one).
    /// Returns the socket path and a join handle yielding the raw
    /// request bytes it read.
    fn fake_server(response: &'static str) -> (PathBuf, std::thread::JoinHandle<Vec<u8>>) {
        let dir = tempfile::tempdir().unwrap();
        // Keep the tempdir alive for the server thread's lifetime by
        // leaking it -- fine for a short-lived test process, and
        // simpler than threading an extra lifetime through the join
        // handle.
        let socket_path = dir.keep().join("cmux.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut reader = BufReader::new(&mut stream);
            let _ = reader.read_until(b'\n', &mut request);
            let mut body = response.as_bytes().to_vec();
            if !body.ends_with(b"\n") {
                body.push(b'\n');
            }
            let _ = stream.write_all(&body);
            request
        });
        (socket_path, handle)
    }

    #[test]
    fn try_show_at_round_trips_a_successful_file_open() {
        let (socket_path, handle) = fake_server(
            r#"{"id":"renderer-show","ok":true,"result":{"surface_id":"11111111-1111-1111-1111-111111111111","pane_id":"22222222-2222-2222-2222-222222222222"}}"#,
        );
        let state_dir = tempfile::tempdir().unwrap();
        let state_file = state_dir.path().join("cmux-preview-surface.json");

        let image_dir = tempfile::tempdir().unwrap();
        let image_path = image_dir.path().join("test.png");
        fs::write(&image_path, b"not-really-a-png-but-unused-here").unwrap();

        let result = try_show_at(&socket_path, &state_file, &image_path);
        assert_eq!(result.unwrap(), Some("cmux"));

        let request = String::from_utf8(handle.join().unwrap()).unwrap();
        let request: serde_json::Value = serde_json::from_str(request.trim()).unwrap();
        assert_eq!(request["method"], "file.open");
        let sent_path = request["params"]["path"].as_str().unwrap();
        assert!(Path::new(sent_path).is_absolute());
        assert_eq!(
            fs::canonicalize(sent_path).unwrap(),
            fs::canonicalize(&image_path).unwrap()
        );

        let saved_state = fs::read_to_string(&state_file).unwrap();
        assert!(saved_state.contains("11111111-1111-1111-1111-111111111111"));
    }

    #[test]
    fn try_show_at_falls_back_when_the_socket_is_unreachable() {
        let missing_socket = tempfile::tempdir().unwrap().path().join("no-such.sock");
        let state_file = tempfile::tempdir()
            .unwrap()
            .path()
            .join("cmux-preview-surface.json");
        let image_dir = tempfile::tempdir().unwrap();
        let image_path = image_dir.path().join("test.png");
        fs::write(&image_path, b"unused").unwrap();

        let result = try_show_at(&missing_socket, &state_file, &image_path);
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn try_show_at_returns_a_clear_error_for_an_ok_false_response() {
        let (socket_path, _handle) = fake_server(
            r#"{"id":"renderer-show","ok":false,"error":{"message":"File not found: /nope.png","code":"not_found"}}"#,
        );
        let state_file = tempfile::tempdir()
            .unwrap()
            .path()
            .join("cmux-preview-surface.json");
        let image_dir = tempfile::tempdir().unwrap();
        let image_path = image_dir.path().join("test.png");
        fs::write(&image_path, b"unused").unwrap();

        let error = try_show_at(&socket_path, &state_file, &image_path).unwrap_err();
        assert_eq!(error.code(), "cmux_error");
        assert!(error.to_string().contains("File not found"));
        assert!(error.to_string().contains("not_found"));
    }

    /// A socket that *accepts a connection* but doesn't answer with
    /// valid JSON-RPC isn't meaningfully different from "cmux isn't
    /// available" -- most plausibly a `CMUX_SOCKET_PATH` value that
    /// leaked (via process-environment inheritance) into a shell
    /// outside the cmux instance that actually owns that socket, so a
    /// live socket answers but isn't really our cmux control channel.
    /// This should fall back to the terminal-protocol path, not hard
    /// error and leave the user with nothing displayed.
    #[test]
    fn try_show_at_falls_back_when_the_response_is_malformed() {
        let (socket_path, _handle) = fake_server("not json at all");
        let state_file = tempfile::tempdir()
            .unwrap()
            .path()
            .join("cmux-preview-surface.json");
        let image_dir = tempfile::tempdir().unwrap();
        let image_path = image_dir.path().join("test.png");
        fs::write(&image_path, b"unused").unwrap();

        let result = try_show_at(&socket_path, &state_file, &image_path);
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn try_show_at_falls_back_when_the_connection_closes_without_a_response() {
        let (socket_path, _handle) = fake_server("");
        let state_file = tempfile::tempdir()
            .unwrap()
            .path()
            .join("cmux-preview-surface.json");
        let image_dir = tempfile::tempdir().unwrap();
        let image_path = image_dir.path().join("test.png");
        fs::write(&image_path, b"unused").unwrap();

        let result = try_show_at(&socket_path, &state_file, &image_path);
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn try_show_at_reports_a_clear_error_when_the_path_cannot_be_canonicalized() {
        // The server is reachable but should never receive a request:
        // canonicalization is checked first and fails here since the
        // path doesn't exist.
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir.path().join("cmux.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        let handle = std::thread::spawn(move || listener.accept());
        let state_file = dir.path().join("cmux-preview-surface.json");

        let error = try_show_at(
            &socket_path,
            &state_file,
            Path::new("/no/such/path/definitely-missing.png"),
        )
        .unwrap_err();

        assert_eq!(error.code(), "io_error");
        assert!(error.to_string().contains("could not read"));
        // No connection should have been made, so the accept thread
        // never returns on its own; drop it without joining.
        drop(handle);
    }

    #[test]
    fn try_clear_at_closes_the_recorded_surface_and_removes_the_state_file() {
        let (socket_path, handle) = fake_server(
            r#"{"id":"renderer-clear","ok":true,"result":{"surface_id":"33333333-3333-3333-3333-333333333333"}}"#,
        );
        let state_dir = tempfile::tempdir().unwrap();
        let state_file = state_dir.path().join("cmux-preview-surface.json");
        fs::write(
            &state_file,
            r#"{"surface_id":"33333333-3333-3333-3333-333333333333"}"#,
        )
        .unwrap();

        let outcome = try_clear_at(&socket_path, &state_file).unwrap();
        assert_eq!(outcome, CmuxClearOutcome::Cleared);
        assert!(!state_file.exists());

        let request = String::from_utf8(handle.join().unwrap()).unwrap();
        let request: serde_json::Value = serde_json::from_str(request.trim()).unwrap();
        assert_eq!(request["method"], "surface.close");
        assert_eq!(
            request["params"]["surface_id"],
            "33333333-3333-3333-3333-333333333333"
        );
    }

    /// A recorded surface can already be gone -- the tab was closed, or
    /// the state file survived from a now-defunct cmux session/window.
    /// `--clear` is meant to be idempotent, so cmux rejecting the close
    /// with a `not_found` code must be treated as success (and the
    /// stale state file cleaned up), not propagated as an error.
    #[test]
    fn try_clear_at_treats_a_not_found_surface_as_already_cleared() {
        let (socket_path, _handle) = fake_server(
            r#"{"id":"renderer-clear","ok":false,"error":{"message":"Workspace not found","code":"not_found"}}"#,
        );
        let state_dir = tempfile::tempdir().unwrap();
        let state_file = state_dir.path().join("cmux-preview-surface.json");
        fs::write(
            &state_file,
            r#"{"surface_id":"33333333-3333-3333-3333-333333333333"}"#,
        )
        .unwrap();

        let outcome = try_clear_at(&socket_path, &state_file).unwrap();
        assert_eq!(outcome, CmuxClearOutcome::NoSurface);
        assert!(!state_file.exists());
    }

    /// A rejection for any other reason must still be a hard error --
    /// only `not_found` is treated as "already cleared".
    #[test]
    fn try_clear_at_still_errors_on_a_non_not_found_rejection() {
        let (socket_path, _handle) = fake_server(
            r#"{"id":"renderer-clear","ok":false,"error":{"message":"internal error","code":"internal"}}"#,
        );
        let state_dir = tempfile::tempdir().unwrap();
        let state_file = state_dir.path().join("cmux-preview-surface.json");
        fs::write(
            &state_file,
            r#"{"surface_id":"33333333-3333-3333-3333-333333333333"}"#,
        )
        .unwrap();

        let error = try_clear_at(&socket_path, &state_file).unwrap_err();
        assert!(error.to_string().contains("internal"));
        // The state file is left alone so a retry has something to act on.
        assert!(state_file.exists());
    }

    #[test]
    fn try_clear_at_reports_no_surface_recorded_without_erroring() {
        let (socket_path, handle) = fake_server("irrelevant: no request is sent");
        let state_file = tempfile::tempdir()
            .unwrap()
            .path()
            .join("cmux-preview-surface.json");

        let outcome = try_clear_at(&socket_path, &state_file).unwrap();
        assert_eq!(outcome, CmuxClearOutcome::NoSurface);

        // A connection is opened (to check availability) but no
        // request is ever sent since there's no surface to close;
        // drop the server's own connection without requiring it to
        // have received anything.
        let _ = handle.join();
    }

    #[test]
    fn try_clear_at_falls_back_when_the_socket_is_unreachable() {
        let missing_socket = tempfile::tempdir().unwrap().path().join("no-such.sock");
        let state_file = tempfile::tempdir()
            .unwrap()
            .path()
            .join("cmux-preview-surface.json");

        let outcome = try_clear_at(&missing_socket, &state_file).unwrap();
        assert_eq!(outcome, CmuxClearOutcome::Unavailable);
    }

    /// Mirrors `try_show_at_falls_back_when_the_response_is_malformed`:
    /// a connectable socket that doesn't answer with valid JSON-RPC is
    /// treated the same as cmux being unavailable, not a hard error.
    #[test]
    fn try_clear_at_falls_back_when_the_response_is_malformed() {
        let (socket_path, _handle) = fake_server("not json at all");
        let state_dir = tempfile::tempdir().unwrap();
        let state_file = state_dir.path().join("cmux-preview-surface.json");
        fs::write(
            &state_file,
            r#"{"surface_id":"33333333-3333-3333-3333-333333333333"}"#,
        )
        .unwrap();

        let outcome = try_clear_at(&socket_path, &state_file).unwrap();
        assert_eq!(outcome, CmuxClearOutcome::Unavailable);
    }
}
