# Ariadnetor

Distributed Tensor Network Framework with pluggable backend architecture.

## Name

**Ariadnetor** is an anagram containing:

- **Ariadne** - The thread through the labyrinth (tensor networks)
- **Tensor** - Multi-dimensional arrays
- **r** - Rust

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  ariadnetor (arnet)  - High-level API                   │
│    Einsum DSL, Expression Graph                         │
├─────────────────────────────────────────────────────────┤
│  ariadnetor-tensor (arnet_tensor)  - CPU Implementation │
│    DenseTensor, FatTensor, Contraction                  │
├─────────────────────────────────────────────────────────┤
│  ariadnetor-core (arnet_core)  - Core Abstractions      │
│    Scalar, LabelId, ComputeBackend trait                │
└─────────────────────────────────────────────────────────┘
```

## Workspace Structure

### `ariadnetor-core`

Backend-agnostic core abstractions.

- `Scalar` / `FloatCompute` traits - Element type abstraction
- `LabelId` - Interned tensor index labels
- `EinsumExpr` / `ContractionPlan` - Einsum parsing and analysis
- `ComputeBackend` trait - Pluggable backend interface

### `ariadnetor-tensor`

Pure Rust CPU tensor implementation.

- `DenseTensor<T>` - Dense storage with Arc-based CoW
- `RawTensor<T>` - Storage format enum (Dense, future: Sparse, BlockSparse)
- `FatTensor<T>` - Tensor with label metadata
- Arithmetic operations (faer for GEMM, hptt for transpose)

### `ariadnetor`

Main library crate (use as `arnet`).

- Einsum DSL
- Expression compute graph
- High-level tensor API

## Prerequisites

- Rust 1.70+

## Installation

```toml
[dependencies]
ariadnetor = { git = "https://github.com/ultimatile/ariadnetor" }
```

## Usage

```rust
use arnet_tensor::{FatTensor, RawTensor, LabelId};

// Create tensors with labels
let a = FatTensor::from_raw(
    RawTensor::<f64>::ones(vec![2, 3]),
    &["i", "j"]
);
let b = FatTensor::from_raw(
    RawTensor::<f64>::ones(vec![3, 4]),
    &["j", "k"]
);

// Contract using einsum notation
let c = a.contract(&b, "ij,jk->ik").unwrap();
```

## Building

```bash
cargo build --workspace
cargo test --workspace
```

### Using cargo-make

```bash
cargo install cargo-make
cargo make build
cargo make test
```

## Roadmap

### Future Backends

These backends are planned but not yet implemented:

- `ariadnetor-cuda` - cuTENSOR / cuTensorNet integration
- `ariadnetor-metal` - Metal Performance Shaders for Apple Silicon
- MLIR/IREE experimentation in a separate repository

## License

MIT OR Apache-2.0
