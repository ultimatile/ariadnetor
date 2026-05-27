use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}
