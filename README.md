# Ariadnetor

> [!WARNING]
> This project is in early development. APIs are unstable and subject to breaking changes.

Tensor network framework in Rust

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
