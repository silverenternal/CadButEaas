# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Rust monolithic CAD geometry intelligent processing system** built on the "Everything-as-a-Service" (EaaS) design philosophy. All functional modules are independent services with clear input/output contracts, composable through uniform interfaces.

- **Version**: v0.1.0
- **Tests**: 585+ tests (584 passing, 1 known failure: `test_polyline_zero_length_edge_filtering`)
- **Clippy**: 0 errors (4 benign complexity warnings)
- **Workspace crates**: 17

---

## Development Commands

### Build

```bash
# Build entire workspace
cargo build --workspace

# Build release with optimizations
cargo build --release

# Build with OpenCV acceleration (4.5x vectorize performance)
cargo build --release --features cad-cli/opencv

# Build specific crate
cargo build -p cad-cli
cargo build -p orchestrator
```

### Test

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p parser
cargo test -p topo

# Run benchmark tests
cargo test --test benchmarks -- --nocapture

# Run E2E tests
cargo test --test e2e_tests

# Run user story tests
cargo test --test user_story_tests

# Run single test by name
cargo test test_halfedge_basic -- --nocapture
```

### Lint & Format

```bash
# Format all code
cargo fmt --workspace

# Clippy check (required: 0 warnings)
cargo clippy --workspace --lib
```

### Run Applications

```bash
# CLI - process file
cargo run --package cad-cli -- process input.dxf --output scene.json

# CLI - with profile
cargo run --package cad-cli -- process input.dxf --profile architectural

# CLI - list/show profiles
cargo run --package cad-cli -- list-profiles
cargo run --package cad-cli -- show-profile architectural

# HTTP server (port 3000)
cargo run --package cad-cli -- serve --port 3000

# GUI viewer (egui)
cargo run --package cad-viewer

# GUI viewer with GPU acceleration
cargo run --package cad-viewer --features gpu
```

### Web Frontend Development (`cad-web/`)

```bash
cd cad-web

# Install dependencies
pnpm install

# Dev server
pnpm dev

# Build
pnpm build

# Lint
pnpm lint
pnpm lint:fix

# Format
pnpm format

# Tests
pnpm test
pnpm test:e2e

# Storybook
pnpm storybook
```

---

## Architecture: Big Picture

### EaaS Design Philosophy

Every module is a **Service** implementing a uniform trait:

```rust
pub trait Service: Send + Sync {
    type Request;
    type Response;
    type Error;
    
    async fn process(&self, request: Self::Request) -> Result<Self::Response, Self::Error>;
}
```

This enables: independent testing, internal evolution without affecting other services, and future microservice deployment without API changes.

### Processing Pipeline (Core Data Flow)

```
                     ┌──────────────────────────────────────────┐
                     │            Orchestrator                   │
                     │         (API Gateway + Pipeline)          │
                     │  HTTP: /health, /process, /acoustic/*     │
                     │  WebSocket: /ws (select edge, gaps, ping) │
                     │  CLI: cad-cli commands                    │
                     └────────────────────┬─────────────────────┘
                                          ↓
┌──────────┐     ┌───────────┐     ┌──────────┐     ┌──────────┐
│  Parser  │ →   │   Topo    │ →   │ Validator│ →   │  Export  │
│ Service  │     │  Service  │     │ Service  │     │ Service  │
│          │     │           │     │          │     │          │
│ DXF/PDF  │     │ R*-tree   │     │ Closure  │     │ JSON     │
│ DWG/SVG  │     │ Intersect │     │ Self-int │     │ Binary   │
│ STL Text │     │ Halfedge  │     │ Holes    │     │ SVG      │
└──────────┘     └─────┬─────┘     └──────────┘     └──────────┘
                        ↓
                  ┌──────────┐     ┌──────────┐
                  │ Vectorize│     │ Acoustic │
                  │  Service │     │ Service  │
                  │          │     │          │
                  │ Image    │     │ Material │
                  │ → Lines  │     │ Reverb   │
                  └──────────┘     └──────────┘

┌───────────────────────────────────────────────────────────────┐
│                         Interact Service                       │
│  Mode A: Edge Picking + Auto Trace                             │
│  Mode B: Lasso/Polygon Selection                               │
│  Gap Detection + Layered Completion (Snap→Bridge→Sem→Manual)   │
│  Boundary Semantic Annotation (HardWall/AbsorptiveWall/etc.)   │
└───────────────────────────────────────────────────────────────┘
```

### Service Call Chain

`process_with_services()` in orchestrator drives the pipeline:

```
1. ParserService::process()     → ParseResult (entities, layers, units)
2. TopoService::process()       → SceneState (outer, holes, boundaries)
3. ValidatorService::process()  → ValidationReport (issues with codes E001-E003, W001-W002)
4. ExportService::process()     → ExportResult (JSON/Binary/SVG bytes)
```

---

## Core Modules Reference

| Crate | Primary Responsibility | Key File |
|-------|------------------------|----------|
| **common-types** | Shared types, errors, geometry primitives | `crates/common-types/src/lib.rs` |
| **parser** | Multi-format file parsing (DXF/DWG/PDF/SVG/STL) | `crates/parser/src/lib.rs` |
| **vectorize** | Raster image vectorization (edge detection → skeletonization → contours) | `crates/vectorize/src/lib.rs` |
| **topo** | Topology construction, Halfedge mesh, R*-tree spatial indexing | `crates/topo/src/lib.rs` |
| **validator** | Geometry quality validation (closure, self-intersect, holes) | `crates/validator/src/lib.rs` |
| **export** | Scene export (JSON, Binary bincode, SVG) | `crates/export/src/lib.rs` |
| **interact** | Interactive collaboration (edge pick, lasso, gap detection) | `crates/interact/src/lib.rs` |
| **orchestrator** | API gateway + pipeline orchestration | `crates/orchestrator/src/lib.rs` |
| **acoustic** | Acoustic analysis (material stats, reverb time) | `crates/acoustic/src/lib.rs` |
| **config** | Profile management (architectural/mechanical/scanned/quick) | `crates/config/src/lib.rs` |
| **cad-cli** | Command-line interface | `crates/cad-cli/src/main.rs` |
| **cad-viewer** | egui-based GUI viewer | `crates/cad-viewer/src/main.rs` |
| **accelerator-*** | Hardware acceleration abstraction | `crates/accelerator-api/src/lib.rs` |
| **raster-loader** | Raster image loading (PNG/JPG/BMP/TIFF/WebP) | `crates/raster-loader/src/lib.rs` |

### Key Algorithm Locations

| Algorithm | Location | Complexity |
|-----------|----------|------------|
| R*-tree endpoint snapping | `crates/topo/src/lib.rs` | O(n log n) |
| Bentley-Ottmann intersection detection | `crates/topo/src/lib.rs` | O((n+k) log n) |
| Halfedge mesh construction | `crates/topo/src/halfedge.rs` | O(n) |
| Douglas-Peucker simplification | `crates/vectorize/src/simplify.rs` | O(n log n) |
| Zhang-Suen skeletonization | `crates/vectorize/src/skeleton.rs` | O(nm) |
| Kåsa circle fitting | `crates/vectorize/src/fitting.rs` | O(n) |

---

## Code Style & Patterns

### Rust Conventions

- **Error handling**: Use `thiserror` for custom error types. No `unwrap()`/`expect()` except tests.
- **Iteration first**: All recursive algorithms must have iterative implementations to avoid stack overflow.
- **Zero-copy**: Large objects use `Arc<T>` for sharing (1000x `Arc::clone()` = 12-13μs).
- **Naming**: PascalCase for types, snake_case for functions/variables.
- **Comments**: Pub items must have doc comments; add comments only when logic is non-obvious.

### Prohibited

- No emoji in code
- No TODO/FIXME comments - implement or don't do it
- No `--no-verify` to skip pre-commit hooks
- No force push to main/master
- No unused abstractions - helpers only created when actually reused

### TypeScript (`cad-web/`)

- Use Biome (not eslint/prettier)
- PascalCase components, camelCase hooks
- Function components only (no class components)
- Tailwind-first, minimal custom CSS

---

## Performance Baselines

Performance regression threshold: **10%** (CI warning), **20%** (CI blocking).

| Module | Metric | Target | Current |
|--------|--------|--------|---------|
| Parser | 1000 entity DXF | <100ms | ✓ |
| Parser | 541,216 entity PDF | <2s | 1.5s |
| Topo | 100 segments | | 13.4ms |
| Topo | 1000 segments | <150ms | 131.9ms |
| Vectorize | 500×500 | | ~50ms |
| Vectorize | 1000×1000 | | ~200ms |
| Vectorize | 2000×2000 | <1s | ~800ms |
| Vectorize (OpenCV) | 2000×3000 | | ~220ms (4.5x faster) |

---

## Test File Locations

- DXF test files: `dxfs/` (9 real architectural drawings)
- PDF test files: `testpdf/` (4 vector PDFs)
- Test suites: `tests/` (e2e_tests.rs, user_story_tests.rs, benchmarks.rs)

---

## Configuration

- Root profile definitions: `cad_config.profiles.toml`
- Example user config: `cad_config.example.toml`
- Config profiles: architectural, mechanical, scanned, quick

---

## Roadmap Context (P2/P3)

Code in these areas may be stub or partially implemented:

- **P2 ✅**: Halfedge structure - now default algorithm, supports nested holes/non-manifold geometry
- **P2 ⏳**: rayon parallelization - dependency present, needs expansion to parser/vectorize
- **P2 ⏳**: PDF vectorization enhancement (dashed/centerline/hatch pattern recognition)
- **P2 ⏳**: Config hot-reload
- **P2 ⏳**: Microservice split (HTTP/gRPC)
- **P3 🔮**: WASM frontend embedding, database integration, OpenTelemetry

---

## Repository Map

```
CAD/
├── Cargo.toml                    # Workspace dependencies
├── cad_config.profiles.toml      # Profile definitions
├── cad_config.example.toml       # Example user config
├── crates/
│   ├── common-types/             # Geometry types, CadError, RecoverySuggestion
│   ├── parser/                   # DXF/DWG/PDF/SVG/STL multi-format parsing
│   ├── vectorize/                # Image→vector conversion (with OpenCV feature)
│   ├── topo/                     # Topology: R*-tree, Halfedge, face enumeration
│   ├── validator/                # Geometry validation (E001-E003, W001-W002)
│   ├── export/                   # JSON, Binary, SVG export
│   ├── interact/                 # Edge picking, lasso, gap detection
│   ├── orchestrator/             # HTTP/WebSocket API + pipeline
│   ├── acoustic/                 # Material statistics, reverb calculation
│   ├── config/                   # Profile management service
│   ├── cad-cli/                  # Command-line interface
│   ├── cad-viewer/               # egui GUI viewer
│   ├── accelerator-api/          # Acceleration abstraction
│   ├── accelerator-cpu/          # CPU implementation
│   ├── accelerator-registry/     # Accelerator service discovery
│   ├── accelerator-wgpu/         # WGPU stub (CPU fallback works)
│   └── raster-loader/            # Raster image format loading
├── cad-web/                      # React + TypeScript frontend
├── dxfs/                         # DXF test files (9 files)
├── testpdf/                      # PDF test files (4 files)
├── tests/                        # Integration/E2E/benchmark tests
├── benches/                      # Benchmark baselines
├── docs/                         # Documentation
└── .github/workflows/ci.yml      # CI/CD pipeline
```
