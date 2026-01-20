//! Error types for tensor contraction operations

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ContractionError {
    InvalidNotation(String),
    LabelMismatch {
        expected: usize,
        actual: usize,
        tensor: String,
    },
    DimensionMismatch {
        label: String,
        lhs_dim: usize,
        rhs_dim: usize,
    },
    LabelNotFound {
        label: String,
        tensor: String,
    },
    DuplicateOutputLabel {
        label: String,
    },
}

impl fmt::Display for ContractionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidNotation(msg) => write!(f, "Invalid Einstein notation: {}", msg),
            Self::LabelMismatch { expected, actual, tensor } => {
                write!(f, "{} tensor: expected {} labels, got {}", tensor, expected, actual)
            }
            Self::DimensionMismatch { label, lhs_dim, rhs_dim } => {
                write!(f, "Dimension mismatch for '{}': lhs={}, rhs={}", label, lhs_dim, rhs_dim)
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
