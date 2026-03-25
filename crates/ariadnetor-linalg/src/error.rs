//! Error types for the linalg layer.

use arnet_core::backend::BackendError;

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
}

impl From<BackendError> for LinalgError {
    fn from(e: BackendError) -> Self {
        Self::Backend(e)
    }
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "Invalid argument: {msg}"),
            Self::Backend(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LinalgError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(e) => Some(e),
            _ => None,
        }
    }
}
