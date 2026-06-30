<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/corona_lockup_dark.png">
    <img src="logo/corona_lockup_light.png" alt="ariadnetor" width="600">
  </picture>
</p>

<p align="center">
  <a href="https://crates.io/crates/ariadnetor"><img src="https://img.shields.io/crates/v/ariadnetor.svg" alt="crates.io"></a>
  <a href="https://docs.rs/ariadnetor"><img src="https://docs.rs/ariadnetor/badge.svg" alt="docs.rs"></a>
  <img src="https://img.shields.io/crates/l/ariadnetor.svg" alt="license">
</p>

> [!WARNING]
> This project is in early development.
> APIs are unstable and subject to breaking changes.

Tensor network library in Rust.

## Installation

```bash
cargo add ariadnetor
```

`ariadnetor` is the component layer (tensors, linear algebra, MPS/MPO).
Tensor-network algorithms (DMRG, …) live in a separate crate:

```bash
cargo add ariadnetor-algorithms
```

### Optional features

Both are off by default, so a plain `cargo add` builds pure-Rust with no C/C++
or system-library dependency.

- **`hptt`** (`ariadnetor`) — routes tensor transposition through the
  [HPTT][hptt] kernel via the [hptt-rs][hptt-rs] bindings. Requires a C++
  compiler and CMake to build HPTT.
- **`arpack`** (`ariadnetor-algorithms`) — adds the [ARPACK-NG][arpack]
  eigensolver backend via the [arpack-rs][arpack-rs] bindings. Requires a
  system ARPACK library discoverable through `pkg-config`.

[hptt]: https://github.com/springer13/hptt
[hptt-rs]: https://github.com/ultimatile/hptt-rs
[arpack]: https://github.com/opencollab/arpack-ng
[arpack-rs]: https://github.com/ultimatile/arpack-rs

See [CONTRIBUTING.md](CONTRIBUTING.md) for build commands and conventions.
