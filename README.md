# Ariadnetor

> [!WARNING]
> This project is in early development. APIs are unstable and subject to breaking changes.

Tensor network framework in Rust

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  ariadnetor (arnet)  - High-level API                   │
│    Tensor, Einsum, MPS/MPO                              │
├──────────────────────────┬──────────────────────────────┤
│  ariadnetor-linalg       │  ariadnetor-native           │
│  (arnet_linalg)          │  (arnet_native)              │
│  Backend-agnostic        │  NativeBackend:              │
│  linear algebra API      │  faer + hptt-rs              │
├──────────────────────────┴──────────────────────────────┤
│  ariadnetor-tensor (arnet_tensor)  - Tensor Data        │
│    Dense, BlockSparse, Sector, TensorRepr               │
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

- `Dense<T>` — zeros, ones, constant, eye, from_data, random, reshape, permute, slice, expand, replace_slice, concatenate, stack, map, conj, to_complex, real, imag, scale, norm, normalize
- `BlockSparse<T, S>` — Block-sparse tensor with abelian symmetry conservation. Stores only flux-allowed blocks in a flat aligned buffer
- `Sector` trait — Abelian symmetry sector algebra (fuse, identity, dual). Implementations: `Z2Sector`, `U1Sector`, tuple products
- `QNIndex<S>` — Quantum-number index: maps sectors to block dimensions with direction (In/Out)
- `TensorRepr` — Common trait for tensor storage representations (`Dense`, `BlockSparse`)
- `Tensor<T, B>` — Main API type: wraps storage + backend

### `ariadnetor-linalg`

Backend-agnostic linear algebra API (via `&impl ComputeBackend`).

- contract, transpose, einsum
- scale, norm, normalize, linear_combine, trace, diag, diagonal_scale
- svd, trunc_svd, qr, lq
- eig, eigh, eigvals, eigvalsh
- expm, expm_hermitian, expm_antihermitian
- solve, inverse
- Block-sparse: contract_block_sparse, svd_block_sparse, trunc_svd_block_sparse, qr_block_sparse, lq_block_sparse

### `ariadnetor-native`

`NativeBackend`: faer + hptt-rs (f32, f64, `Complex<f32>`, `Complex<f64>`)

### `ariadnetor`

Main library crate (`arnet`). Re-exports + high-level API (`arnet::ops`).

### `ariadnetor-mps`

Consumer crate above the tensor library (`arnet_mps`). MPS/MPO tensor chains: canonicalize, truncate, inner product, braket, MPO application, site operators (SpinHalf, Qubit). Add as a direct dependency alongside `ariadnetor`.

## Usage

```rust
use arnet::{Dense, Tensor, contract, svd};

// Create tensors
let a = Tensor::<Dense<f64>>::zeros(vec![2, 3]);
let b = Tensor::<Dense<f64>>::zeros(vec![3, 2]);

// Tensor contraction
let c = contract(&a, &b, "ij,jk->ik").unwrap();

// SVD decomposition
let (u, s, vt) = svd(&a, 1).unwrap();
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
