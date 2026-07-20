//! Term-list handling for the SRC linear combination: validation of the
//! conditions the shared sketch imposes, zero-coefficient pruning, and
//! the coefficient-weighted panel/site combiner.

use ariadnetor_core::Scalar;
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensor, linear_combine};
use num_traits::{Float, One, Zero};

use super::super::super::chain::TensorChain;
use super::super::super::types::SumTerm;

/// One term of a linear combination, instantiated at the dense flavor.
pub(crate) type DenseTerm<'a, T> = SumTerm<'a, DenseStorage<T>, DenseLayout>;

/// Validate the term list against the conditions the shared sketch and
/// panel summation impose, returning the common chain length. Within a
/// term the MPO ket leg must contract with the MPS physical leg; across
/// terms only the MPO output (bra) dimensions must agree — that is where
/// the shared Gaussian acts and where the summed panels' rows must line
/// up. Input spaces may differ per term. Runs before any contraction so a
/// malformed term list fails here, not inside an einsum. Every term is
/// validated, including zero-weighted ones a later pruning pass drops —
/// disabling a term must not hide a malformed operand.
pub(super) fn validate_terms<T: Scalar>(terms: &[DenseTerm<'_, T>], coeffs: &[T]) -> usize {
    assert!(!terms.is_empty(), "terms must be non-empty");
    assert_eq!(coeffs.len(), terms.len(), "one coefficient per term");
    assert!(
        coeffs
            .iter()
            .all(|c| c.re().is_finite() && c.im().is_finite()),
        "coefficients must be finite"
    );
    let n = terms[0].1.len();
    assert!(n > 0, "must have at least one site");
    for (op, psi) in terms {
        assert_eq!(op.len(), n, "all terms' chains must have equal length");
        assert_eq!(psi.len(), n, "all terms' chains must have equal length");
    }
    for i in 0..n {
        let d_out = terms[0].0.site(i).shape()[2];
        for (op, psi) in terms {
            let w = op.site(i);
            let a = psi.site(i);
            assert_eq!(
                w.shape()[1],
                a.shape()[1],
                "MPO ket and MPS physical dimensions must match within a term"
            );
            assert_eq!(
                w.shape()[2],
                d_out,
                "MPO output (bra) dimensions must agree across terms"
            );
        }
    }
    n
}

/// Drop zero-weighted terms (the reference implementation prunes them the
/// same way). A zero coefficient must behave as "term absent": flowing the
/// term through the combiner would let a non-finite element in a disabled
/// term poison the summed panel (`0 * inf = NaN` surfacing as a spurious
/// `NonFinite` error), and its environments would cost sweep work for a
/// contribution that is identically zero. An all-zero result (empty
/// vectors) means the sum is the zero state; the caller owns that
/// representation.
pub(super) fn prune_zero_terms<'a, T: Scalar>(
    terms: &[DenseTerm<'a, T>],
    coeffs: &[T],
) -> (Vec<DenseTerm<'a, T>>, Vec<T>) {
    let is_zero = |c: T| c.re() == T::Real::zero() && c.im() == T::Real::zero();
    terms
        .iter()
        .zip(coeffs.iter())
        .filter(|(_, c)| !is_zero(**c))
        .map(|(&t, &c)| (t, c))
        .unzip()
}

/// Coefficient-weighted sum of per-term tensors. A lone coefficient-one
/// term is returned untouched: `linear_combine` starts from zero and
/// evaluates `0 + c * x`, which is not bit-identical to `x` under signed
/// zero and complex-product rounding, and the single-term wrapper's
/// contract is exact equality with the pre-generalization kernel — any
/// panel difference could flip an adaptive decision and change later
/// block sizes.
pub(super) fn weighted_sum<T: Scalar>(
    mut tensors: Vec<DenseTensor<T>>,
    coeffs: &[T],
) -> DenseTensor<T> {
    debug_assert_eq!(tensors.len(), coeffs.len());
    let is_one = |c: T| c.re() == T::Real::one() && c.im() == T::Real::zero();
    if tensors.len() == 1 && is_one(coeffs[0]) {
        return tensors.pop().expect("length checked above");
    }
    let refs: Vec<&DenseTensor<T>> = tensors.iter().collect();
    linear_combine(&refs, coeffs).expect(
        "weighted sum: per-term operands share shape (validated dimensions, shared cap rank) \
             and memory order (same contraction pipeline)",
    )
}
