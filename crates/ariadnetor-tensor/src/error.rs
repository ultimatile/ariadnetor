use thiserror::Error;

/// Errors raised by tensor construction and manipulation.
#[derive(Debug, Error)]
pub enum TensorError {
    /// An argument violated a precondition of the called operation; the
    /// payload describes which argument and why.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}
