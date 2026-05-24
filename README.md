# Ariadnetor

> [!WARNING]
> This project is in early development. APIs are unstable and subject to breaking changes.

Tensor network framework in Rust

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  ariadnetor-algorithms (arnet_algorithms)               │
│    DMRG, TEBD                                           │
├─────────────────────────────────────────────────────────┤
│  ariadnetor-mps (arnet_mps)                             │
│    MPS/MPO chains, site ops, canonicalize, truncate     │
├─────────────────────────────────────────────────────────┤
│  ariadnetor (arnet) — tensor library umbrella           │
│    Re-exports the layers below                          │
├─────────────────────────────────────────────────────────┤
│  ariadnetor-linalg (arnet_linalg)                       │
│    Backend-agnostic linear algebra (&Tensor in / out)   │
├─────────────────────────────────────────────────────────┤
│  ariadnetor-tensor (arnet_tensor)                       │
│    Tensor, DenseTensor, BlockSparseTensor, Sector       │
├─────────────────────────────────────────────────────────┤
│  ariadnetor-native (arnet_native)                       │
│    NativeBackend: faer + hptt-rs                        │
├─────────────────────────────────────────────────────────┤
│  ariadnetor-core (arnet_core)                           │
│    ComputeBackend trait, Scalar, MemoryOrder, EinsumExpr│
└─────────────────────────────────────────────────────────┘
```

Each layer depends only on the layers below it. `ariadnetor` (the
umbrella) re-exports `core` + `native` + `tensor` + `linalg`, so most
downstream code can consume the tensor library through a single `arnet`
dependency. `ariadnetor-mps` and `ariadnetor-algorithms` sit above the
umbrella as separate consumer crates.

## Workspace Structure

### `ariadnetor-core`

Backend-agnostic core abstractions: `Scalar` trait, `LabelId`, `EinsumExpr`, `ComputeBackend` trait.

### `ariadnetor-tensor`

User-facing tensor types with Arc-based Copy-on-Write storage.

- `Tensor<St, L, B>` — joins a `TensorData<St, L>` storage / layout bundle with an `Arc<B>` compute backend.
- `DenseTensor<T, B = NativeBackend>` — dense `Tensor` alias. Constructors: `zeros`, `ones`, `constant`, `eye`, `random`, `zeros_with_backend`, `from_raw_parts`. Methods: `conj`, `scale`, `norm`, `normalize`, `reordered`, `order`, `shape`, etc.
- `BlockSparseTensor<T, S, B = NativeBackend>` — block-sparse `Tensor` alias with abelian symmetry. Constructors: `zeros`, `zeros_with_backend`, `random`, `from_block_fn`, `from_raw_parts`. Methods: `dagger`, `conj`, `norm`, `order`, `block_data`, `flux`, `indices`.
- `Sector` trait — abelian symmetry sector algebra (fuse, identity, dual). Implementations: `U1Sector`, `Z2Sector`, tuple products.
- `QNIndex<S>` — quantum-number index: maps sectors to block dimensions with direction (`In` / `Out`).

`Dense<T>` / `BlockSparse<T, S>` are the underlying storage primitives kept `pub` for cross-crate kernel access; user code should reach for `DenseTensor` / `BlockSparseTensor` instead.

### `ariadnetor-linalg`

Backend-agnostic linear algebra. Public functions take `&Tensor` and return `Tensor`; the backend flows from the tensor argument.

- contract, transpose, einsum
- scale, norm, normalize, linear_combine, trace, diag, diagonal_scale
- svd, trunc_svd, qr, lq
- eig, eigh, eigvals, eigvalsh
- expm, expm_hermitian, expm_antihermitian
- solve, inverse
- Block-sparse: contract_block_sparse, svd_block_sparse, trunc_svd_block_sparse, qr_block_sparse, lq_block_sparse

### `ariadnetor-native`

`NativeBackend`: faer + hptt-rs (f32, f64, `Complex<f32>`, `Complex<f64>`).

### `ariadnetor`

Umbrella crate (`arnet`). Re-exports `arnet_core`, `arnet_native`, `arnet_tensor`, and `arnet_linalg` so most downstream code can depend on a single crate. Does not re-export `arnet_mps` or `arnet_algorithms`; consumers of MPS / DMRG add those as separate dependencies.

### `ariadnetor-mps`

MPS/MPO tensor chains (`arnet_mps`): `Mps` / `Mpo` constructors with per-chain order assertion, site accessors, `canonicalize`, `truncate`, `inner`, `braket`, `apply`, site operators (`SpinHalf`, `Qubit`). Add as a direct dependency.

### `ariadnetor-algorithms`

Tensor-network algorithms (`arnet_algorithms`): DMRG (`sweep_2site`, `DmrgEnvs`, effective-Hamiltonian solvers), Krylov internals (`lanczos`, optional `arpack` feature). Add as a direct dependency.

## Usage

```rust
use arnet::{DenseTensor, contract};

// Create tensors
let a = DenseTensor::<f64>::zeros(vec![2, 3]);
let b = DenseTensor::<f64>::zeros(vec![3, 2]);

// Tensor contraction
let c = contract(&a, &b, "ij,jk->ik").unwrap();
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
