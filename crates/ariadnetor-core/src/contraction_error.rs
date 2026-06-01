//! Error types for tensor contraction operations

/// Error raised while parsing or validating an Einstein-notation contraction.
///
/// Every variant fully describes its own failure; none wraps a structured
/// inner error. `ContractionError` is therefore a leaf in the error chain —
/// its `source()` is always `None`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ContractionError {
    #[error("Invalid Einstein notation: {0}")]
    InvalidNotation(String),
    #[error("{tensor} tensor: expected {expected} labels, got {actual}")]
    LabelMismatch {
        expected: usize,
        actual: usize,
        tensor: String,
    },
    #[error("Dimension mismatch for '{label}': lhs={lhs_dim}, rhs={rhs_dim}")]
    DimensionMismatch {
        label: String,
        lhs_dim: usize,
        rhs_dim: usize,
    },
    #[error("Label '{label}' not found in {tensor} tensor")]
    LabelNotFound { label: String, tensor: String },
    #[error("Duplicate label '{label}' in output")]
    DuplicateOutputLabel { label: String },
}
