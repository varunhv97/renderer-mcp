# Repository Guidelines

## Workspace layout

This repository is planned as a Rust workspace. Keep shared and deployable
components under `crates/`:

- `schema` for versioned scene-schema types and validation.
- `renderer` for the wgpu rendering engine.
- `daemon` for the local rendering service.
- `cli` for command-line workflows.
- `mcp` for MCP integration.

Place end-to-end and cross-crate coverage in `tests/`. Keep fixtures and visual
test inputs in `assets/`. Treat `.renderer/` as generated local output; do not
commit its contents unless a future repository policy explicitly says otherwise.

## Development commands

Once the workspace exists, run these checks before opening a pull request:

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo llvm-cov --workspace --all-targets --all-features --fail-under-lines 65
```

Use the relevant package commands during development, for example
`cargo run -p cli -- --help` for the CLI and `cargo run -p mcp` for the MCP
server. Update these examples when package names or invocation contracts change.

## Rust conventions

- Format with `rustfmt` and use four-space indentation.
- Name modules and functions in `snake_case`; name types and traits in
  `PascalCase`.
- Prefer descriptive, domain-specific error enums over opaque string errors.
- Give scene schemas explicit, versioned names and preserve compatibility rules
  in validation and migration code.

## Testing expectations

- Keep focused unit tests colocated with the code they exercise.
- Put integration and cross-crate scenarios under `tests/`.
- Use deterministic fixture names such as `diagram_basic.scene.json`.
- Add perceptual golden-image tests for rendering behavior when they provide
  stable, meaningful coverage; document any approved tolerance.
- Maintain at least 65% line coverage across the workspace. Run `cargo llvm-cov`
  with `--fail-under-lines 65` locally and enforce the same command in CI.
  Increase this threshold as the renderer gains a portable GPU test backend.

## Commits and pull requests

Use Conventional Commit-style subjects, for example:

```text
feat: add scene validator
```

Keep pull requests focused. Include a concise description, test evidence, the
linked issue when one exists, and visual artifacts or screenshots for rendering
changes.

## Security and input handling

- Resolve assets locally only; do not introduce implicit remote asset fetching.
- Never place secrets, credentials, or tokens in fixtures or generated output.
- Treat scene and shader input as untrusted: validate it before parsing,
  compiling, or rendering, and constrain resource use where practical.
