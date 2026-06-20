//! Error types for tensor contraction operations

/// Error raised while parsing or validating an Einstein-notation contraction.
///
/// Every variant fully describes its own failure; none wraps a structured
/// inner error. `ContractionError` is therefore a leaf in the error chain —
/// its `source()` is always `None`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ContractionError {
    /// The notation string could not be parsed as valid Einstein notation.
    #[error("Invalid Einstein notation: {0}")]
    InvalidNotation(String),
    /// A tensor's label count does not match its rank.
    #[error("{tensor} tensor: expected {expected} labels, got {actual}")]
    LabelMismatch {
        /// Number of labels expected (the tensor's rank).
        expected: usize,
        /// Number of labels actually supplied.
        actual: usize,
        /// Which operand the mismatch occurred on.
        tensor: String,
    },
    /// A shared label is bound to different extents on the two operands.
    #[error("Dimension mismatch for '{label}': lhs={lhs_dim}, rhs={rhs_dim}")]
    DimensionMismatch {
        /// The label whose extents disagree.
        label: String,
        /// Extent of the label on the left operand.
        lhs_dim: usize,
        /// Extent of the label on the right operand.
        rhs_dim: usize,
    },
    /// A referenced label is absent from the named tensor.
    #[error("Label '{label}' not found in {tensor} tensor")]
    LabelNotFound {
        /// The label that was not found.
        label: String,
        /// Which operand was searched.
        tensor: String,
    },
    /// The output specification repeats a label.
    #[error("Duplicate label '{label}' in output")]
    DuplicateOutputLabel {
        /// The label that appears more than once in the output.
        label: String,
    },
}
