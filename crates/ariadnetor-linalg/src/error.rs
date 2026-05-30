//! Error types for the linalg layer.

use arnet_core::backend::BackendError;
use arnet_tensor::TensorError;

/// Error from a linalg operation.
///
/// Separates linalg-layer argument validation from backend-originated errors.
/// The `Backend` / `Tensor` variants are pure repackages of their child error
/// types: they add no layer context, so they forward `Display` and `source()`
/// to the inner error via `#[error(transparent)]`. Keeping the cause out of the
/// wrapper's own `Display` avoids surfacing it twice in a `source()`-walking
/// reporter (e.g. `anyhow`'s `{:#}`).
#[derive(Debug, thiserror::Error)]
pub enum LinalgError {
    /// Argument validation failed in the linalg layer.
    ///
    /// The backend was never called. Examples: nrow out of range,
    /// non-square matrix where square is required, shape mismatch.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// The backend reported an error during execution.
    #[error(transparent)]
    Backend(#[from] BackendError),

    /// An error raised by an underlying `arnet-tensor` operation.
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn from_tensor_error_wraps_in_tensor_variant() {
        let inner = TensorError::InvalidArgument("x".to_string());
        let err: LinalgError = inner.into();
        match err {
            LinalgError::Tensor(TensorError::InvalidArgument(msg)) => assert_eq!(msg, "x"),
            other => panic!("expected Tensor variant, got {other:?}"),
        }
    }

    #[test]
    fn tensor_variant_display_delegates_to_inner() {
        let err: LinalgError = TensorError::InvalidArgument("y".to_string()).into();
        assert_eq!(err.to_string(), "Invalid argument: y");
    }

    #[test]
    fn tensor_variant_source_is_transparent_to_inner() {
        // Transparent wrap: `source()` exposes the inner's source, not the
        // inner itself. `TensorError::InvalidArgument` has no inner source,
        // so the wrap's `source()` is `None`.
        let err: LinalgError = TensorError::InvalidArgument("z".to_string()).into();
        assert!(err.source().is_none());
    }

    #[test]
    fn from_backend_error_wraps_in_backend_variant() {
        let inner = BackendError::ExecutionFailed("boom".to_string());
        let err: LinalgError = inner.into();
        match err {
            LinalgError::Backend(BackendError::ExecutionFailed(msg)) => assert_eq!(msg, "boom"),
            other => panic!("expected Backend variant, got {other:?}"),
        }
    }

    #[test]
    fn backend_variant_display_delegates_to_inner() {
        // `#[error(transparent)]` forwards `Display` to the inner error
        // verbatim — the wrapper adds no text of its own. The distinctive
        // "Execution failed:" prefix (vs the "Invalid argument:" the other
        // variants share) makes the delegation observable.
        let err: LinalgError = BackendError::ExecutionFailed("boom".to_string()).into();
        assert_eq!(err.to_string(), "Execution failed: boom");
    }

    #[test]
    fn backend_variant_source_is_transparent_to_inner() {
        // `BackendError` is a leaf (empty `Error` impl), so the transparent
        // wrap's `source()` forwards to the inner's `source()` of `None`.
        // This is the deliberate behavior change from the previous
        // hand-written impl, which returned `Some(inner)` here and so
        // double-printed the cause under a `source()`-walking reporter.
        let err: LinalgError = BackendError::ExecutionFailed("boom".to_string()).into();
        assert!(err.source().is_none());
    }

    #[test]
    fn transparent_wrapper_does_not_duplicate_cause() {
        // Non-duplication contract: a transparent wrapper's `Display` is
        // exactly the inner's `Display`, so a reporter that prints the
        // wrapper and then walks `source()` cannot surface the cause twice.
        let inner = BackendError::ExecutionFailed("distinctive-marker".to_string());
        let inner_display = inner.to_string();
        let err: LinalgError = inner.into();
        assert_eq!(err.to_string(), inner_display);
        assert!(err.source().is_none());
    }
}
