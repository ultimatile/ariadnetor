//! Error types for tensor contraction operations

use std::fmt;

/// Error type for contraction operations
#[derive(Debug, Clone, PartialEq)]
pub enum ContractionError {
    /// Invalid Einstein notation
    InvalidNotation(String),

    /// Label count mismatch between notation and tensor
    LabelMismatch {
        expected: usize,
        actual: usize,
        tensor: String, // "lhs" or "rhs"
    },

    /// Dimension mismatch for contracted labels
    DimensionMismatch {
        label: String,
        lhs_dim: usize,
        rhs_dim: usize,
    },

    /// Label not found in tensor
    LabelNotFound {
        label: String,
        tensor: String, // "lhs" or "rhs"
    },

    /// Duplicate labels in output
    DuplicateOutputLabel {
        label: String,
    },
}

impl fmt::Display for ContractionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidNotation(msg) => {
                write!(f, "Invalid Einstein notation: {}", msg)
            }
            Self::LabelMismatch {
                expected,
                actual,
                tensor,
            } => {
                write!(
                    f,
                    "{} tensor: expected {} labels in notation, got {} labels in tensor",
                    tensor, expected, actual
                )
            }
            Self::DimensionMismatch {
                label,
                lhs_dim,
                rhs_dim,
            } => {
                write!(
                    f,
                    "Dimension mismatch for contracted label '{}': lhs dim = {}, rhs dim = {}",
                    label, lhs_dim, rhs_dim
                )
            }
            Self::LabelNotFound { label, tensor } => {
                write!(f, "Label '{}' not found in {} tensor", label, tensor)
            }
            Self::DuplicateOutputLabel { label } => {
                write!(f, "Duplicate label '{}' in output", label)
            }
        }
    }
}

impl std::error::Error for ContractionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ContractionError::InvalidNotation("missing arrow".to_string());
        assert!(err.to_string().contains("Invalid Einstein notation"));

        let err = ContractionError::LabelMismatch {
            expected: 2,
            actual: 3,
            tensor: "lhs".to_string(),
        };
        assert!(err.to_string().contains("lhs tensor"));
        assert!(err.to_string().contains("expected 2"));

        let err = ContractionError::DimensionMismatch {
            label: "j".to_string(),
            lhs_dim: 3,
            rhs_dim: 4,
        };
        assert!(err.to_string().contains("Dimension mismatch"));
        assert!(err.to_string().contains("'j'"));
    }
}
