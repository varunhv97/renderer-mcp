# Development

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo llvm-cov --workspace --all-targets --all-features --fail-under-lines 65
```

`cargo llvm-cov` needs `llvm-tools`. If your toolchain came from `rustup`,
`rustup component add llvm-tools-preview` is enough. On a Homebrew-installed
stable toolchain (no `rustup`), point it at a matching LLVM instead:

```sh
brew install llvm
LLVM_COV="$(brew --prefix llvm)/bin/llvm-cov" \
LLVM_PROFDATA="$(brew --prefix llvm)/bin/llvm-profdata" \
cargo llvm-cov --workspace --all-targets --all-features --fail-under-lines 65
```

Building produces two binaries: `renderer` (CLI, package `renderer-cli`) and
`renderer-mcp` (MCP server, package `renderer-mcp`).

```sh
cargo build --release
```


## Testing

Unit tests are colocated with each crate; cross-crate CLI and MCP protocol
tests live under `crates/cli/tests` and `crates/mcp/tests`. The renderer crate
also includes deterministic golden-image tests (fixtures under
`crates/renderer/assets/golden/`) that render representative scenes and diff
the output against checked-in reference PNGs within a documented pixel
tolerance; like the other GPU-backed tests, they skip rather than fail on a
host with no GPU adapter. CI enforces formatting, Clippy, the full test suite,
and a 65% workspace line-coverage floor via `cargo llvm-cov` on macOS, Linux,
and Windows.
