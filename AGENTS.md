# Repository Guidelines

## Project Structure & Module Organization

The Rust workspace is organized under `crates/`, with services such as `parser`, `vectorize`, `topo`, `validator`, `export`, and `orchestrator`; shared contracts live in `common-types` and `service-kit`. Cross-crate tests and benchmarks are in `tests/` and `benches/`. The React/TypeScript client lives in `cad-web/src`, with tests in `cad-web/tests`. Python CadStruct-MoE utilities are under `scripts/vlm`; generated outputs belong in `reports/`, `datasets/`, or `checkpoints/`. Consult `ARCHITECTURE.md` before changing service boundaries.

## Build, Test, and Development Commands

- `cargo build --workspace --all-targets` builds all Rust crates and test targets.
- `cargo test --workspace --all-targets` runs the full Rust suite; use `cargo test -p topo` for one crate.
- `cargo fmt --all` formats Rust code; `cargo clippy --workspace --all-targets` performs static checks.
- `cd cad-web && npm install && npm run dev` starts the Vite frontend locally.
- `cd cad-web && npm run build` type-checks and creates a production build.
- `cd cad-web && npm test` runs Vitest; `npm run test:e2e` runs Playwright.
- `pytest` runs Python tests; target a file for faster iteration, such as `pytest tests/test_floorplancad_panoptic_window_merge.py`.

## Coding Style & Naming Conventions

Use `rustfmt` and Rust 2021 idioms. Name types in `PascalCase`, functions and modules in `snake_case`, and constants in `SCREAMING_SNAKE_CASE`. Return structured `Result` errors instead of panicking in libraries, and document public APIs. Frontend code uses TypeScript, functional React components, and Biome (`npm run lint`, `npm run format`); component files use `PascalCase`, while hooks start with `use`.

## Testing Guidelines

Place Rust unit tests beside implementations and integration tests in `tests/*.rs`. Name Python tests `test_*.py` and frontend tests `.test.ts` or `.test.tsx`. Add regression coverage for fixes and prefer small fixtures over generated artifacts. No fixed coverage percentage is enforced; test changed behavior at the narrowest appropriate layer.

## Commit & Pull Request Guidelines

History primarily follows Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, and `chore:`. Keep subjects imperative and focused. Pull requests should explain motivation, link issues, list validation commands, and include screenshots for UI changes. Call out configuration, model, dataset, or benchmark impacts, and ensure relevant checks pass before review.

## Security & Configuration

Start local configuration from `cad_config.example.toml`. Never commit credentials, private datasets, or machine-specific paths. Avoid adding large model weights or generated checkpoints unless the change explicitly requires and documents them.
