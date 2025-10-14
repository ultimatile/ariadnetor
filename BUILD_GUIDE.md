# TN-MLIR Build Guide

This guide explains how to build and integrate the TN-Compute MLIR dialect.

## Current Status

✅ **Completed**:
- TableGen dialect definitions (TNBase.td, TNTypes.td, TNOps.td)
- C++ implementation (TNDialect.cpp, TNOps.cpp)
- Lowering passes (TN → LinAlg)
- Rust scaffolding (dialect, builder, jit, tensor modules)
- Basic tests and examples

⏳ **Remaining**:
- MLIR/LLVM installation and configuration
- Building C++ dialect with CMake
- Rust-C++ integration via FFI
- JIT compiler implementation
- Runtime function implementations (SLATE/Lamellar backends)

---

## Prerequisites

### 1. Install LLVM/MLIR

#### Option A: Build from source (recommended for development)

```bash
# Clone LLVM project
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout release/19.x  # Use LLVM 19.x

# Configure with MLIR enabled
cmake -G Ninja -B build -S llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local \
  -DLLVM_ENABLE_ASSERTIONS=ON

# Build (takes ~1-2 hours)
ninja -C build

# Install
ninja -C build install
```

#### Option B: Use system package (if available)

```bash
# macOS (Homebrew)
brew install llvm@19

# Ubuntu/Debian
apt-get install llvm-19-dev mlir-19-tools libmlir-19-dev

# Set environment variable
export LLVM_SYS_190_PREFIX=/path/to/llvm-19
```

---

## Building the TN-Compute Dialect

### Step 1: Configure CMake

```bash
cd tn-mlir

# Configure
cmake -B build -S . \
  -DMLIR_DIR=$HOME/.local/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/.local/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local
```

### Step 2: Build

```bash
cmake --build build -j$(nproc)
```

This will generate:
- `build/lib/libMLIRTNDialect.a` - TN dialect library
- `build/lib/libMLIRTNTransforms.a` - Transform passes
- `build/include/tn-compute/Dialect/IR/*.h.inc` - Generated headers

### Step 3: Install

```bash
cmake --install build
```

---

## Integrating with Rust

### Step 1: Enable MLIR feature

Edit `Cargo.toml`:

```toml
[dependencies]
melior = { version = "0.19", optional = false }  # Enable melior

[features]
default = ["mlir"]
mlir = ["melior"]
```

### Step 2: Update build.rs

Implement the build script to:
1. Detect MLIR installation
2. Link against TN dialect libraries
3. Link against MLIR/LLVM libraries

```rust
// build.rs
use std::env;
use std::path::PathBuf;

fn main() {
    // Check for MLIR installation
    let mlir_dir = env::var("MLIR_DIR")
        .expect("MLIR_DIR environment variable not set");

    let project_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let tn_mlir_lib = format!("{}/build/lib", project_dir);

    // Link TN dialect libraries
    println!("cargo:rustc-link-search=native={}", tn_mlir_lib);
    println!("cargo:rustc-link-lib=static=MLIRTNDialect");
    println!("cargo:rustc-link-lib=static=MLIRTNTransforms");

    // Link MLIR libraries
    let mlir_lib = format!("{}/lib", mlir_dir);
    println!("cargo:rustc-link-search=native={}", mlir_lib);

    for lib in [
        "MLIRIR", "MLIRSupport", "MLIRLinalgDialect",
        "MLIRTensorDialect", "MLIRArithDialect",
        "MLIRFuncDialect", "MLIRTransforms"
    ] {
        println!("cargo:rustc-link-lib=static={}", lib);
    }
}
```

### Step 3: Create FFI bindings

Create `src/ffi.rs`:

```rust
//! FFI bindings to TN-Compute MLIR dialect

use std::os::raw::c_void;

#[repr(C)]
pub struct MlirContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MlirDialect {
    _private: [u8; 0],
}

extern "C" {
    // Dialect registration
    pub fn mlirGetDialectHandle__tn__() -> *mut c_void;
    pub fn mlirDialectHandleRegisterDialect(
        handle: *mut c_void,
        context: *mut MlirContext,
    );

    // TODO: Add more FFI functions for operations
}
```

### Step 4: Implement dialect registration

Update `src/dialect.rs`:

```rust
use melior::Context;
use crate::ffi;

pub struct TNDialect {
    context: Context,
}

impl TNDialect {
    pub fn new() -> Result<Self> {
        let context = Context::new();

        unsafe {
            let handle = ffi::mlirGetDialectHandle__tn__();
            ffi::mlirDialectHandleRegisterDialect(
                handle,
                context.as_raw_ptr() as *mut _,
            );
        }

        Ok(Self { context })
    }
}
```

---

## Implementing JIT Compilation

### Step 1: Use melior ExecutionEngine

Update `src/jit.rs`:

```rust
use melior::{
    Context, ExecutionEngine, Module,
    dialect::DialectRegistry,
    ir::{Block, Location, Region},
    pass::{PassManager, Pass},
};

pub struct TNJITCompiler {
    context: Context,
    engine: ExecutionEngine,
}

impl TNJITCompiler {
    pub fn new() -> Result<Self> {
        let context = Context::new();

        // Register dialects
        let registry = DialectRegistry::new();
        // ... register TN, LinAlg, etc.

        // Create execution engine
        let engine = ExecutionEngine::new(&module, 2, &[], false);

        Ok(Self { context, engine })
    }

    pub fn compile_and_execute(
        &mut self,
        einsum_expr: &str,
        tensors: Vec<Tensor>,
    ) -> Result<Tensor> {
        // 1. Build TN-Compute IR
        let module = self.build_module(einsum_expr, &tensors)?;

        // 2. Run optimization passes
        self.optimize(&module)?;

        // 3. JIT compile
        self.engine.compile(&module)?;

        // 4. Execute
        let result_ptr = unsafe {
            self.engine.invoke_packed("main", &tensor_ptrs)
        };

        Ok(Tensor::from_ptr(result_ptr))
    }

    fn build_module(&self, expr: &str, tensors: &[Tensor]) -> Result<Module> {
        // Parse einsum notation
        // Build TN dialect operations
        // ...
    }

    fn optimize(&self, module: &Module) -> Result<()> {
        let pm = PassManager::new(&self.context);

        // Add passes
        pm.add_pass(Pass::create_convert_tn_to_linalg());
        pm.add_pass(Pass::create_linalg_fusion());
        pm.add_pass(Pass::create_convert_linalg_to_llvm());

        pm.run(module)?;
        Ok(())
    }
}
```

---

## Runtime Function Implementation

### Option A: SLATE Backend

Create runtime functions that call SLATE:

```rust
// src/runtime/slate.rs

#[no_mangle]
pub extern "C" fn tn_runtime_svd(
    input_ptr: *const f64,
    m: i64, n: i64,
    max_chi: i64,
    threshold: f64,
) -> (*const f64, *const f64, *const f64) {
    // Call SLATE gesvd
    let result = slate::gesvd(input_ptr, m, n);

    // Apply truncation if needed
    if max_chi > 0 || threshold > 0.0 {
        result = truncate_svd(result, max_chi, threshold);
    }

    // Return pointers to U, S, V
    (result.u_ptr(), result.s_ptr(), result.v_ptr())
}

#[no_mangle]
pub extern "C" fn tn_runtime_qr(
    input_ptr: *const f64,
    m: i64, n: i64,
) -> (*const f64, *const f64) {
    // Call SLATE geqrf
    let result = slate::geqrf(input_ptr, m, n);
    (result.q_ptr(), result.r_ptr())
}
```

### Option B: Lamellar Backend

For distributed execution with Lamellar:

```rust
// src/runtime/lamellar.rs

use lamellar::prelude::*;

#[no_mangle]
pub extern "C" fn tn_runtime_distributed_svd(
    tensor_id: usize,
    max_chi: i64,
    threshold: f64,
) -> usize {
    let world = get_lamellar_world();

    // Execute distributed SVD task
    let result_id = world.exec_am_all(DistributedSVDTask {
        tensor_id,
        max_chi,
        threshold,
    }).block();

    result_id
}
```

---

## Testing

### Unit Tests

```bash
cargo test
```

### Integration Tests

Create `tests/mlir_integration.rs`:

```rust
#[test]
fn test_matrix_multiplication() {
    let mut compiler = TNJITCompiler::new().unwrap();

    let a = Tensor::ones(vec![10, 20]);
    let b = Tensor::ones(vec![20, 30]);

    let c = compiler.compile_and_execute("ij,jk->ik", vec![a, b]).unwrap();

    assert_eq!(c.shape(), &[10, 30]);
    assert_eq!(c.get(&[0, 0]), 20.0);  // sum of 20 ones
}

#[test]
fn test_svd() {
    let mut compiler = TNJITCompiler::new().unwrap();

    let a = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2]
    );

    // Build SVD operation
    let builder = TNBuilder::new().unwrap();
    let (u, s, v) = builder.svd(&a.data(), Some(2), None).unwrap();

    // Verify decomposition: A ≈ U * S * V^T
    // ...
}
```

### Benchmark

```bash
cargo bench
```

---

## Troubleshooting

### LLVM version mismatch

```
error: failed to find correct version (19.x.x) of llvm-config
```

**Solution**: Set `LLVM_SYS_190_PREFIX` environment variable:

```bash
export LLVM_SYS_190_PREFIX=/path/to/llvm-19
```

### Linking errors

```
undefined reference to `mlir::tn::TNDialect::initialize()'
```

**Solution**: Ensure build.rs links against all required libraries.

### Runtime errors

```
JIT compilation failed: unknown dialect 'tn'
```

**Solution**: Verify dialect is registered before creating operations.

---

## Next Steps

1. **Week 1-2**: Complete MLIR installation and C++ dialect build
2. **Week 3-4**: Implement Rust FFI bindings and dialect registration
3. **Week 5-6**: Implement operation builders with melior
4. **Week 7-8**: Implement JIT compiler and runtime functions
5. **Week 9+**: Begin PoC A (SLATE) and PoC B (Lamellar) in parallel

---

## References

- [MLIR Getting Started](https://mlir.llvm.org/getting_started/)
- [melior Documentation](https://docs.rs/melior)
- [LLVM Rust Bindings](https://gitlab.com/taricorp/llvm-sys.rs)
- [Design Documents](../docs/)
