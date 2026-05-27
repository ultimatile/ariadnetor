//! Error types for the linalg layer.

use arnet_core::backend::BackendError;
use arnet_tensor::TensorError;

/// Error from a linalg operation.
///
/// Separates linalg-layer argument validation from backend-originated errors.
/// Backend errors propagate through the `Backend` variant via the `From` impl.
#[derive(Debug)]
pub enum LinalgError {
    /// Argument validation failed in the linalg layer.
    ///
    /// The backend was never called. Examples: nrow out of range,
    /// non-square matrix where square is required, shape mismatch.
    InvalidArgument(String),

    /// The backend reported an error during execution.
    Backend(BackendError),

    /// An error raised by an underlying `arnet-tensor` operation.
    Tensor(TensorError),
}

impl From<BackendError> for LinalgError {
    fn from(e: BackendError) -> Self {
        Self::Backend(e)
    }
}

impl From<TensorError> for LinalgError {
    fn from(e: TensorError) -> Self {
        Self::Tensor(e)
    }
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "Invalid argument: {msg}"),
            Self::Backend(e) => write!(f, "{e}"),
            Self::Tensor(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LinalgError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(e) => Some(e),
            // Transparent wrap: expose the inner's source, not the inner
            // itself. Since `Display` already delegates to the inner, also
            // returning `Some(e)` would surface the same message twice in
            // an `Error::source()`-walking reporter.
            Self::Tensor(e) => e.source(),
            _ => None,
        }
    }
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
}
