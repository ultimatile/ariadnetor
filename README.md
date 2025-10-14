# TN-MLIR: Tensor Network MLIR Dialect

A Rust-based distributed tensor network library with MLIR compilation frontend.

## Project Structure

```
tn-mlir/
├── include/tn-compute/Dialect/IR/   # MLIR Dialect definitions
│   ├── TNBase.td                     # Dialect base
│   ├── TNTypes.td                    # Type system
│   ├── TNOps.td                      # Operations (TableGen)
│   ├── TNDialect.h                   # Dialect header
│   └── Transforms/
│       └── Passes.h                  # Transformation passes
├── lib/Dialect/
│   ├── IR/
│   │   ├── TNDialect.cpp            # Dialect implementation
│   │   └── TNOps.cpp                # Operation verifiers
│   └── Transforms/
│       └── ConvertTNToLinalg.cpp    # TN → LinAlg lowering
├── src/                             # Rust implementation
│   ├── lib.rs                       # Main library
│   ├── dialect.rs                   # Dialect registration
│   ├── builder.rs                   # Operation builders
│   └── jit.rs                       # JIT compiler
└── CMakeLists.txt                   # MLIR build configuration
```

## Architecture

```
┌─────────────────────────────────────────┐
│ MLIR Frontend                            │
│ ┌─────────────────────────────────────┐ │
│ │ DSL Parser (Einsum notation)         │ │
│ │  ↓                                   │ │
│ │ Tensor Dialect                       │ │
│ │  ↓                                   │ │
│ │ TN-Compute Dialect                   │ │← Custom implementation
│ │  ↓                                   │ │
│ │ Optimization passes                  │ │
│ │  ↓                                   │ │
│ │ LinAlg Dialect                       │ │
│ └─────────────────────────────────────┘ │
└─────────────────┬───────────────────────┘
                  ↓ JIT/AOT Compilation
┌─────────────────────────────────────────┐
│ Execution Runtime                        │
│ (Distributed linear algebra backends)    │
└─────────────────────────────────────────┘
```

## TN-Compute Dialect Operations

### Tensor Contractions

```mlir
// Matrix multiplication
%C = tn.contract %A, %B {
    indices = "ij,jk->ik"
} : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>

// High-dimensional contraction
%result = tn.contract %T1, %T2 {
    indices = "ijkl,klmn->ijmn"
} : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64> -> tensor<?x?x?x?xf64>
```

### Matrix Decompositions

```mlir
// Singular Value Decomposition
%U, %S, %V = tn.svd %A {
    max_chi = 100,
    threshold = 1.0e-10
} : tensor<?x?xf64> -> (tensor<?x?xf64>, tensor<?xf64>, tensor<?x?xf64>)

// QR Decomposition
%Q, %R = tn.qr %A : tensor<?x?xf64> -> (tensor<?x?xf64>, tensor<?x?xf64>)
```

### Tensor Manipulations

```mlir
// Transpose
%transposed = tn.transpose %A {
    permutation = [1, 0]
} : tensor<?x?xf64> -> tensor<?x?xf64>

// Reshape
%reshaped = tn.reshape %A {
    new_shape = [100, 10]
} : tensor<1000xf64> -> tensor<100x10xf64>

// Bond dimension truncation
%truncated = tn.truncate %tensor {
    max_chi = 50,
    threshold = 1.0e-8
} : tensor<?x?xf64> -> tensor<?x?xf64>
```

## Building the MLIR Dialect

### Prerequisites

- LLVM/MLIR 17+ installed
- CMake 3.20+
- C++17 compiler

### Build Steps

```bash
# Configure
cmake -B build -S . \
  -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm

# Build
cmake --build build

# Install
cmake --install build --prefix ~/.local
```

### Generated Files

TableGen will generate the following files:
- `TNOps.h.inc` / `TNOps.cpp.inc` - Operation definitions
- `TNDialect.h.inc` / `TNDialect.cpp.inc` - Dialect definitions
- `TNTypes.h.inc` / `TNTypes.cpp.inc` - Type definitions

## Rust Integration (melior)

### Dependencies

```toml
[dependencies]
melior = "0.19"
llvm-sys = "170"
```

### Example Usage

```rust
use tn_mlir::{TNJITCompiler, Tensor};

fn main() {
    let mut compiler = TNJITCompiler::new();

    // Compile einsum expression
    let a = Tensor::new(vec![100, 200]);
    let b = Tensor::new(vec![200, 300]);

    let c = compiler.compile_and_execute(
        "ij,jk->ik",
        vec![a, b]
    );

    println!("Result shape: {:?}", c.shape());
}
```

## Lowering Passes

### TN → LinAlg

Operations are lowered as follows:

| TN Operation | Target |
|--------------|--------|
| `tn.contract` (matrix multiply) | `linalg.matmul` |
| `tn.contract` (general) | `linalg.generic` |
| `tn.transpose` | `linalg.transpose` |
| `tn.svd` | Runtime function call |
| `tn.qr` | Runtime function call |

### Runtime Functions

SVD and QR operations require external linear algebra libraries and are
lowered to runtime function calls:

```mlir
// Before lowering
%U, %S, %V = tn.svd %A

// After lowering
%U, %S, %V = func.call @tn_runtime_svd(%A, %max_chi, %threshold)
```

These runtime functions are implemented using distributed linear algebra libraries.

## References

- [MLIR Documentation](https://mlir.llvm.org/)
- [melior (Rust MLIR bindings)](https://docs.rs/melior)

## License

TBD
