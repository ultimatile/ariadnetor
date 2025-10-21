# Ariadnetor

MLIR-based Distributed Tensor Network Framework written in Rust.

## Name

**Ariadnetor** is an anagram containing:

- **Ariadne** - The thread through the labyrinth (tensor networks)
- **IR** - Intermediate Representation (MLIR)
- **Tensor** - Multi-dimensional arrays
- **r** - Rust

## Structure

This is a workspace containing multiple crates:

### `ariadnetor-tensor-dialect`

MLIR Tensor Dialect for tensor operations.

- Dialect definition (C++/TableGen)
- Lowering passes (TN → LinAlg)
- IR Builder (Rust)
- MemRef descriptors for FFI

### `ariadnetor`

Main library crate (use as `arnet`).

- Einsum DSL
- Tensor API
- Runtime functions (faer for GEMM, hptt for transpose)

## Installation

```toml
[dependencies]
ariadnetor = { git = "https://github.com/ultimatile/ariadnetor" }
```

## Usage

```rust
use arnet::{Tensor, einsum};

// Matrix multiplication using einsum notation
let a = Tensor::new(vec![100, 200]);
let b = Tensor::new(vec![200, 300]);

let c = einsum("ij,jk->ik", vec![&a, &b]);
```

## Building

```bash
cargo build --workspace
```

## License

MIT OR Apache-2.0
