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

### `ariadnetor-tensor-compute-dialect`

MLIR Tensor Compute Dialect for tensor operations.

- Dialect definition (C++/TableGen)
- Lowering passes (TC → LinAlg)
- IR Builder (Rust)
- MemRef descriptors for FFI

### `ariadnetor`

Main library crate (use as `arnet`).

- Einsum DSL
- Tensor API
- Runtime functions (faer for GEMM, hptt for transpose)

## Prerequisites

- Rust 1.70+
- **LLVM/MLIR 20.x (required)** - Other versions are not compatible with melior 0.25
- CMake 3.20+

### Installing LLVM/MLIR

**Important**: You must use LLVM 20.x. Other versions (19.x, 21.x) will cause build or runtime errors due to melior compatibility.

```bash
# macOS
brew install llvm@20

# Set environment variable
export MLIR_SYS_200_PREFIX=/opt/homebrew/opt/llvm@20
```

See [LLVM Getting Started](https://llvm.org/docs/GettingStarted.html) for building from source.

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

## Troubleshooting

### LLVM version mismatch error

If you see errors like `failed to find correct version of llvm-config` or runtime crashes:

- Ensure you have **LLVM 20.x** installed (not 19.x or 21.x)
- Set the environment variable: `export MLIR_SYS_200_PREFIX=/path/to/llvm-20`
- Verify with: `$MLIR_SYS_200_PREFIX/bin/llvm-config --version` (should show 20.x.x)

### Linking errors

If you encounter undefined symbol errors during linking:

- Ensure `MLIR_SYS_200_PREFIX` points to the correct LLVM 20.x installation
- On macOS: `export MLIR_SYS_200_PREFIX=/opt/homebrew/opt/llvm@20`

## License

MIT OR Apache-2.0
