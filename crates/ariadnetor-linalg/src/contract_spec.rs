//! Shared parse + validation for the two-operand `contract` inputs, in both the
//! notation and axis-pair forms.
//!
//! Both the dense and block-sparse layouts route their `contract` through this
//! module so they accept and reject exactly the same notation. The supported
//! shape is a two-operand tensordot with free output reordering: every input
//! label is either a single output label or a contracted label (shared by both
//! operands and omitted from the output). Partial trace (a repeated label within
//! one operand), implicit reduction (a label in one operand only, absent from
//! the output), batch indices (a label in both operands and the output), and
//! any operand count other than two are rejected here with `InvalidArgument`
//! rather than reaching a downstream panic.
//!
//! The axis-pair face of the same contraction — the `tensordot` entry, where the
//! caller names contracted axes directly instead of via labels — shares the
//! range / duplicate / arity checks through [`validate_contraction_axes_pair`],
//! so the dense and block-sparse axis kernels reject identical malformed axes
//! with one message.

use std::collections::HashSet;

use ariadnetor_core::EinsumExpr;

use crate::error::LinalgError;

/// Validate that `expr` is a well-formed two-operand contraction.
///
/// Rejects, with `InvalidArgument`: any operand count other than two; a label
/// repeated within either operand or within the output; a batch index (present
/// in both operands and the output); and an input label that is neither
/// contracted nor present in the output. This is the single rejection contract
/// shared by the dense and block-sparse kernels.
pub(crate) fn validate_contract_notation(expr: &EinsumExpr) -> Result<(), LinalgError> {
    if expr.num_inputs() != 2 {
        return Err(LinalgError::InvalidArgument(format!(
            "contract requires exactly two operands; notation has {}",
            expr.num_inputs()
        )));
    }

    let lhs = expr.lhs_indices();
    let rhs = expr.rhs_indices();
    let out = expr.out_indices();

    reject_repeat(lhs, "left operand")?;
    reject_repeat(rhs, "right operand")?;
    reject_repeat(out, "output")?;

    let lhs_set: HashSet<u8> = lhs.iter().copied().collect();
    let rhs_set: HashSet<u8> = rhs.iter().copied().collect();
    let out_set: HashSet<u8> = out.iter().copied().collect();

    // Batch index: shared by both operands and kept in the output. Hadamard /
    // batched contraction belongs in einsum, not contract.
    for &c in out {
        if lhs_set.contains(&c) && rhs_set.contains(&c) {
            return Err(LinalgError::InvalidArgument(format!(
                "contract() does not support batch index '{}'; use einsum() instead",
                c as char
            )));
        }
    }

    // Every input label must be either contracted (shared, omitted) or a free
    // output label. A label in one operand only and absent from the output is an
    // implicit reduction, which contract does not perform.
    for &c in lhs {
        if !rhs_set.contains(&c) && !out_set.contains(&c) {
            return Err(implicit_reduction_err(c, "left operand"));
        }
    }
    for &c in rhs {
        if !lhs_set.contains(&c) && !out_set.contains(&c) {
            return Err(implicit_reduction_err(c, "right operand"));
        }
    }

    Ok(())
}

/// Parsed two-operand contraction: contracted axis pairs plus the natural and
/// requested output label orders.
///
/// `axes_lhs[k]` / `axes_rhs[k]` are the operand axis positions of the `k`-th
/// contracted label, paired by shared label. `natural_labels` is the output leg
/// order a plain tensordot produces — free left labels in left-operand axis
/// order, then free right labels in right-operand axis order. `out_labels` is
/// the order the notation requests; when it differs from `natural_labels` the
/// caller permutes the tensordot result into it.
pub(crate) struct ContractSpec {
    pub(crate) axes_lhs: Vec<usize>,
    pub(crate) axes_rhs: Vec<usize>,
    pub(crate) natural_labels: Vec<u8>,
    pub(crate) out_labels: Vec<u8>,
    /// Number of indices the notation assigns to the left / right operand. The
    /// caller checks these against the actual operand ranks (a notation naming
    /// fewer axes than the operand has would otherwise silently treat the
    /// undeclared axes as free).
    pub(crate) lhs_arity: usize,
    pub(crate) rhs_arity: usize,
}

impl ContractSpec {
    /// Parse and validate `notation`, deriving the contracted axis pairs and the
    /// natural / requested output label orders.
    pub(crate) fn from_notation(notation: &str) -> Result<Self, LinalgError> {
        let expr = EinsumExpr::parse(notation)
            .map_err(|e| LinalgError::InvalidArgument(format!("Failed to parse einsum: {e}")))?;
        validate_contract_notation(&expr)?;

        let lhs = expr.lhs_indices();
        let rhs = expr.rhs_indices();
        let out_labels = expr.out_indices().to_vec();

        let lhs_set: HashSet<u8> = lhs.iter().copied().collect();
        let rhs_set: HashSet<u8> = rhs.iter().copied().collect();

        // Contracted pairs, paired by shared label in left-operand axis order so
        // both operands' k-th contracted axis carry the same label.
        let mut axes_lhs = Vec::new();
        let mut axes_rhs = Vec::new();
        for (i, &c) in lhs.iter().enumerate() {
            if rhs_set.contains(&c) {
                axes_lhs.push(i);
                axes_rhs.push(
                    rhs.iter()
                        .position(|&x| x == c)
                        .expect("shared label present in rhs"),
                );
            }
        }

        // Natural tensordot output order: free left labels (left axis order),
        // then free right labels (right axis order). Validation guarantees a
        // label absent from the other operand is a free output label.
        let mut natural_labels = Vec::with_capacity(out_labels.len());
        for &c in lhs {
            if !rhs_set.contains(&c) {
                natural_labels.push(c);
            }
        }
        for &c in rhs {
            if !lhs_set.contains(&c) {
                natural_labels.push(c);
            }
        }

        Ok(Self {
            axes_lhs,
            axes_rhs,
            natural_labels,
            out_labels,
            lhs_arity: lhs.len(),
            rhs_arity: rhs.len(),
        })
    }
}

/// Validate a set of contraction axis pairs: `axes_lhs` and `axes_rhs` must have
/// equal length, every axis must be in range for its operand rank, and no axis
/// may repeat within either list. Shared by the dense and block-sparse axis-pair
/// contraction kernels (the block-sparse kernel layers its quantum-number
/// compatibility checks on top).
///
/// Messages are operand-neutral — they carry no `tensordot:` prefix — because
/// the block-sparse `contract` kernel reaches this check on the axis pairs it
/// derives from a parsed notation, not only the direct `tensordot` entry.
pub(crate) fn validate_contraction_axes_pair(
    axes_lhs: &[usize],
    lhs_rank: usize,
    axes_rhs: &[usize],
    rhs_rank: usize,
) -> Result<(), LinalgError> {
    if axes_lhs.len() != axes_rhs.len() {
        return Err(LinalgError::InvalidArgument(format!(
            "axes_lhs length {} != axes_rhs length {}",
            axes_lhs.len(),
            axes_rhs.len()
        )));
    }
    reject_out_of_range_or_duplicate(axes_lhs, lhs_rank, "axes_lhs")?;
    reject_out_of_range_or_duplicate(axes_rhs, rhs_rank, "axes_rhs")?;
    Ok(())
}

fn reject_out_of_range_or_duplicate(
    axes: &[usize],
    rank: usize,
    which: &str,
) -> Result<(), LinalgError> {
    for (i, &a) in axes.iter().enumerate() {
        if a >= rank {
            return Err(LinalgError::InvalidArgument(format!(
                "{which}[{i}] = {a} out of range for rank {rank}"
            )));
        }
        if axes[..i].contains(&a) {
            return Err(LinalgError::InvalidArgument(format!(
                "{which} contains duplicate axis {a}"
            )));
        }
    }
    Ok(())
}

fn reject_repeat(indices: &[u8], where_: &str) -> Result<(), LinalgError> {
    let mut seen = HashSet::new();
    for &c in indices {
        if !seen.insert(c) {
            return Err(LinalgError::InvalidArgument(format!(
                "contract() does not support a repeated index '{}' within the {where_}; \
                 use a dedicated trace for partial traces",
                c as char
            )));
        }
    }
    Ok(())
}

fn implicit_reduction_err(c: u8, where_: &str) -> LinalgError {
    LinalgError::InvalidArgument(format!(
        "index '{}' in the {where_} is neither contracted nor in the output; \
         contract() does not perform implicit reduction",
        c as char
    ))
}

#[cfg(test)]
mod tests;
