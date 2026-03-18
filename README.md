# Ariadnetor

> [!WARNING]
> This project is in early development. APIs are unstable and subject to breaking changes.

Tensor network framework in Rust

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  ariadnetor (arnet)  - High-level API                   │
│    Einsum DSL, Expression Graph, Runtime                │
├──────────────────────────┬──────────────────────────────┤
│  ariadnetor-linalg       │  ariadnetor-native           │
│  (arnet_linalg)          │  (arnet_native)              │
│  Backend-agnostic        │  NativeBackend:              │
│  linear algebra API      │  faer + hptt-rs              │
├──────────────────────────┴──────────────────────────────┤
│  ariadnetor-tensor (arnet_tensor)  - Tensor Data        │
│    DenseTensor, TensorStorage, Tensor                   │
├─────────────────────────────────────────────────────────┤
│  ariadnetor-core (arnet_core)  - Core Abstractions      │
│    Scalar, LabelId, ComputeBackend trait, EinsumExpr    │
└─────────────────────────────────────────────────────────┘
```

## Workspace Structure

### `ariadnetor-core`

Backend-agnostic core abstractions: `Scalar` trait, `LabelId`, `EinsumExpr`, `ComputeBackend` trait.

### `ariadnetor-tensor`

Tensor data structures with Arc-based Copy-on-Write.

- `DenseTensor<T>` — zeros, ones, constant, eye, from_data, random, reshape, permute, slice, expand, replace_slice, concatenate, stack, map, conj, to_complex, real, imag
- `TensorStorage<T>` — Storage format enum (Dense)
- `Tensor<T>` — Main API type: scale, linear_combine, norm, normalize

### `ariadnetor-linalg`

Backend-agnostic linear algebra API (via `&impl ComputeBackend`).

- contract, transpose
- scale, norm, normalize, linear_combine, trace, diag
- svd, trunc_svd, qr, lq
- eig, eigh, eigvals, eigvalsh
- expm, expm_hermitian, expm_antihermitian
- solve, inverse

### `ariadnetor-native`

`NativeBackend`: faer + hptt-rs (f32, f64, `Complex<f32>`, `Complex<f64>`)

### `ariadnetor`

Main library crate (`arnet`). Re-exports + `ExpressionComputeGraph` (evaluate not yet implemented).

## Usage

```rust
use arnet::{Tensor, NativeBackend};
use arnet_linalg::{contract, svd, transpose};

// Create tensors
let a = Tensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
let b = Tensor::<f64>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

// Tensor contraction via ComputeBackend
let backend = NativeBackend::new();
let c = contract(&a.storage, &b.storage, "ij,jk->ik", &backend).unwrap();

// SVD decomposition
let result = svd(&a.storage, 0, &backend).unwrap();
```

## Building

```bash
cargo make build       # Build workspace
cargo make test        # Run unit tests
cargo make ci          # Full CI checks (fmt, clippy, test)
```

Or with plain cargo:

```bash
cargo build --workspace
cargo test --workspace
```

## Prerequisites

- Rust (edition 2024)
- [hptt-rs](https://github.com/ultimatile/hptt-rs) (for high-performance transpose)

## License

MIT OR Apache-2.0
