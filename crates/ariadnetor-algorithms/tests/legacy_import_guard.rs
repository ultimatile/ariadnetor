//! Compiles the shared legacy-import guard against this crate's
//! production sources. The single guard source lives in the sibling
//! mps crate; its `env!`-based paths (`CARGO_MANIFEST_DIR`) expand
//! against the crate compiling the test, so the included scan covers
//! this crate's `src/`.
#[path = "../../ariadnetor-mps/tests/legacy_import_guard.rs"]
mod legacy_import_guard;
