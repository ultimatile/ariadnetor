//! Incremental (column-block-append) thin QR decomposition.
//!
//! [`IncrementalQr`] grows a thin QR factorization one column block at a
//! time without refactorizing the columns already absorbed: each append
//! runs block classical Gram-Schmidt with one reorthogonalization pass
//! (BCGS2). The single extra projection is what makes the accumulated
//! basis orthonormal to working precision in the well-conditioned case —
//! one pass alone already degrades with the block's condition number.
//!
//! Optionally the inverse triangular factor is maintained across appends
//! through the block-triangular inverse identity
//! `[[R, C], [0, R22]]^-1 = [[G, -G C G22], [0, G22]]` (with `G = R^-1`,
//! `G22 = R22^-1`), so the squared row norms of `R^-1` — the quantity
//! randomized leave-one-out error estimators consume — update in
//! `O(p^2 s)` per append instead of the `O(p^3)` full inversion a
//! from-scratch recompute pays every round. The maintained inverse needs
//! no refresh pass: the row-norm functional the estimators consume is
//! dominated by its well-conditioned rows, which the incremental update
//! computes to full accuracy, so it tracks a from-scratch inversion far
//! inside the estimators' own stochastic spread even next to the rank
//! tolerance.
//!
//! The accumulated basis is NOT unconditionally orthonormal, in two ways.
//! A rank-deficient append (detected from the maintained diagonal of `R`,
//! terminating the factorization) can overlap the existing span through
//! its Householder completion columns. And even without a
//! [`QrAppendOutcome::RankDeficient`] outcome, orthogonality loss can
//! compound across many appends whose projected parts are ill-conditioned
//! — a regime the diagonal test cannot see, because the offending
//! diagonal entries sit legitimately above the tolerance. The terminal
//! accessor [`IncrementalQr::into_orthonormal_q`] therefore repairs the
//! basis with one plain [`qr`](crate::qr) pass whenever more than a
//! single block was appended.
//!
//! Dense-only: the consumer is the randomized MPO-MPS compression sweep,
//! whose Gaussian sketch has no block-sparse counterpart.

use ariadnetor_core::Scalar;
use ariadnetor_tensor::{
    DenseStorage, DenseTensor, DenseTensorData, OpsFor, add_all, linear_combine,
};
use num_traits::{Float, NumCast, One, Zero};

use crate::error::LinalgError;
use crate::{inverse_with_backend, qr, tensordot};

#[cfg(test)]
mod tests;

/// Result of one [`IncrementalQr::append`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QrAppendOutcome {
    /// The appended block passed the rank test, so it extended the
    /// factorization and the inverse state (when tracked) was updated.
    /// This is not an orthonormality claim about the accumulated basis —
    /// only [`IncrementalQr::into_orthonormal_q`] guarantees that.
    FullRank,
    /// The maintained diagonal of `R` failed the rank test after the
    /// append. The block's columns were still absorbed so the span stays
    /// complete (the orthonormality repair happens in
    /// [`IncrementalQr::into_orthonormal_q`]), and the factorization is
    /// terminated: the inverse state is left untouched and further
    /// appends panic.
    RankDeficient,
}

/// Incrementally grown thin QR factorization of a logically stacked
/// matrix `[B_1 B_2 ...]` of appended column blocks.
///
/// The struct owns only the factor state; the backend performing the
/// kernels is an argument of each method.
pub struct IncrementalQr<T: Scalar> {
    nrows: usize,
    track_r_inverse: bool,
    /// Accumulated basis, `nrows x ncols`; orthonormal only to whatever
    /// precision the appends achieved (see the module doc), which is why
    /// [`Self::into_orthonormal_q`], not this field, is what callers get.
    /// `None` until the first append.
    q: Option<DenseTensor<T>>,
    /// Maintained `R^-1`, `ncols x ncols` upper triangular. `None` unless
    /// `track_r_inverse` and at least one full-rank append happened.
    g: Option<DenseTensor<T>>,
    /// `|R_ii|` across every append, for the rank test. Its length is the
    /// column count.
    r_diag: Vec<T::Real>,
    /// Squared row norms of `g`, kept in step with it.
    row_inv_sq: Vec<T::Real>,
    /// Number of committed appends; decides whether
    /// [`Self::into_orthonormal_q`] needs its repair pass.
    appends: usize,
    terminated: bool,
}

impl<T: Scalar> IncrementalQr<T> {
    /// Empty factorization over `nrows`-row column blocks.
    ///
    /// `track_r_inverse` selects whether appends maintain `R^-1` and its
    /// row norms (needed by [`Self::r_inverse_row_sq_norms`]); skipping it
    /// spares the per-append inversion when only `Q` is wanted.
    ///
    /// # Panics
    ///
    /// Panics if `nrows` is zero.
    pub fn new(nrows: usize, track_r_inverse: bool) -> Self {
        assert!(nrows >= 1, "nrows must be at least 1");
        Self {
            nrows,
            track_r_inverse,
            q: None,
            g: None,
            r_diag: Vec::new(),
            row_inv_sq: Vec::new(),
            appends: 0,
            terminated: false,
        }
    }

    /// Number of columns absorbed so far.
    pub fn ncols(&self) -> usize {
        self.r_diag.len()
    }

    /// Squared row norms of the maintained `R^-1`, one per column.
    ///
    /// `None` when inverse tracking is off, when nothing has been
    /// appended yet, or when the factorization is terminated (a
    /// rank-deficient append leaves the inverse state stale by design —
    /// a singular block has no inverse to fold in).
    pub fn r_inverse_row_sq_norms(&self) -> Option<&[T::Real]> {
        // `g` exists only when tracking is on and a full-rank append
        // happened, so it subsumes the tracking flag.
        (!self.terminated && self.g.is_some()).then_some(self.row_inv_sq.as_slice())
    }

    /// Consume the factorization and return an orthonormal basis whose
    /// span contains every appended column (`nrows x ncols`; after a
    /// rank-deficient final append the basis is wider than the data's
    /// span, exactly like the thin QR of the stacked rank-deficient
    /// matrix).
    ///
    /// A single append is one plain Householder QR, whose Q is orthonormal
    /// by construction and returned as is. Any longer history gets one
    /// re-orthonormalizing [`qr`](crate::qr) pass, which repairs both
    /// gradual block Gram-Schmidt loss and the overlap a rank-deficient
    /// final append can introduce (see the module doc), while keeping
    /// every appended column inside the span.
    ///
    /// # Errors
    ///
    /// Propagates backend failures of the re-orthonormalizing QR.
    ///
    /// # Panics
    ///
    /// Panics if nothing was appended.
    pub fn into_orthonormal_q<B: OpsFor<DenseStorage<T>>>(
        self,
        backend: &B,
    ) -> Result<DenseTensor<T>, LinalgError> {
        let q = self
            .q
            .expect("into_orthonormal_q requires at least one appended block");
        if self.appends == 1 {
            return Ok(q);
        }
        let (q, _) = qr(backend, &q, 1)?;
        Ok(q)
    }

    /// Append a column block (`nrows x s` with `s >= 1`) to the
    /// factorization.
    ///
    /// # Errors
    ///
    /// Returns [`LinalgError::InvalidArgument`] when the block is not a
    /// matrix of `nrows` rows, is empty, would push the column count past
    /// `nrows` (the triangular factor must stay square for the rank test
    /// and the inverse update), or comes from a backend whose memory order
    /// differs from the one the factorization was started with — a single
    /// factorization must be driven by a single backend, since its stored
    /// factors are assembled from the kernels' output buffers. Failures of
    /// the underlying backend kernels also propagate. In every `Err` case
    /// the factorization state is unchanged.
    ///
    /// # Panics
    ///
    /// Panics when called after an append returned
    /// [`QrAppendOutcome::RankDeficient`].
    pub fn append<B: OpsFor<DenseStorage<T>>>(
        &mut self,
        backend: &B,
        block: &DenseTensor<T>,
    ) -> Result<QrAppendOutcome, LinalgError> {
        assert!(
            !self.terminated,
            "append on a terminated IncrementalQr (a prior append returned RankDeficient)"
        );
        let shape = block.shape();
        if shape.len() != 2 || shape[0] != self.nrows {
            return Err(LinalgError::InvalidArgument(format!(
                "block must be a matrix with {} rows, got shape {:?}",
                self.nrows, shape
            )));
        }
        let s = shape[1];
        if s == 0 {
            return Err(LinalgError::InvalidArgument(
                "block must have at least one column".to_string(),
            ));
        }
        if let Some(q) = &self.q {
            let order = backend.preferred_order();
            if q.data().order() != order {
                return Err(LinalgError::InvalidArgument(format!(
                    "backend produces {order:?} but the factorization holds {:?}; \
                     one IncrementalQr must be driven by a single backend",
                    q.data().order()
                )));
            }
        }
        let p_old = self.ncols();
        if p_old + s > self.nrows {
            return Err(LinalgError::InvalidArgument(format!(
                "appending {s} columns to {p_old} would exceed the {} rows; \
                 R would no longer be square",
                self.nrows
            )));
        }

        // Orthogonalize the block against the existing basis (BCGS2), or
        // factorize it directly when the state is empty. `C` accumulates
        // both projection passes; it is the off-diagonal block of the
        // grown triangular factor.
        let (q2, r22, c) = match &self.q {
            None => {
                let (q2, r22) = qr(backend, block, 1)?;
                (q2, r22, None)
            }
            Some(q) => {
                // `Scalar` carries no `Neg`; -1 is built through the real
                // component instead.
                let minus_one = T::one().scale_real(-T::Real::one());
                let q_conj = q.conj();
                let c1 = tensordot(backend, &q_conj, block, &[0], &[0])?;
                let proj1 = tensordot(backend, q, &c1, &[1], &[0])?;
                let b1 = linear_combine(&[block, &proj1], &[T::one(), minus_one])?;
                let c2 = tensordot(backend, &q_conj, &b1, &[0], &[0])?;
                let proj2 = tensordot(backend, q, &c2, &[1], &[0])?;
                let b_perp = linear_combine(&[&b1, &proj2], &[T::one(), minus_one])?;
                let c = add_all(&[&c1, &c2])?;
                let (q2, r22) = qr(backend, &b_perp, 1)?;
                (q2, r22, Some(c))
            }
        };

        // Rank test on the candidate diagonal before committing anything:
        // the new columns can also demote an old diagonal entry by raising
        // the maximum, and a deficient factor must not be inverted.
        let mut new_diag = Vec::with_capacity(s);
        for i in 0..s {
            new_diag.push(r22.get([i, i]).abs());
        }
        let p_new = p_old + s;
        let mut max_diag = T::Real::zero();
        for d in self.r_diag.iter().chain(new_diag.iter()) {
            if *d > max_diag {
                max_diag = *d;
            }
        }
        let p_real = <T::Real as NumCast>::from(p_new).expect("column counts fit in the real type");
        let tol = T::Real::epsilon() * p_real * max_diag;
        let deficient = self.r_diag.iter().chain(new_diag.iter()).any(|d| *d <= tol);

        // Remaining fallible work runs before any state mutation, so an
        // `Err` from any path leaves the factorization exactly as it was.
        let inverse_update = if self.track_r_inverse && !deficient {
            let g22 = inverse_with_backend(backend, &r22, 1)?;
            let x = match (&self.g, &c) {
                (Some(g), Some(c)) => {
                    // X = -G C G22, the off-diagonal block of the grown
                    // inverse; old rows gain its row norms, new rows are
                    // rows of G22.
                    let u = tensordot(backend, g, c, &[1], &[0])?;
                    Some(tensordot(backend, &u, &g22, &[1], &[0])?.scaled(-T::Real::one()))
                }
                (None, None) => None,
                // The tracked inverse exists iff a prior append exists,
                // which is exactly when the projection block was computed.
                _ => unreachable!("inverse state and projection block always appear together"),
            };
            Some((g22, x))
        } else {
            None
        };

        // Commit; nothing below can fail.
        self.appends += 1;
        self.r_diag.extend(new_diag);
        let q_new = match self.q.take() {
            None => q2,
            Some(q) => {
                DenseTensor::from_data(DenseTensorData::concatenate(&[q.data(), q2.data()], 1))
            }
        };
        self.q = Some(q_new);

        if deficient {
            self.terminated = true;
            return Ok(QrAppendOutcome::RankDeficient);
        }

        if let Some((g22, x)) = inverse_update {
            match x {
                None => {
                    for i in 0..s {
                        self.row_inv_sq.push(row_sq_norm(&g22, i));
                    }
                    self.g = Some(g22);
                }
                Some(x) => {
                    let g = self
                        .g
                        .take()
                        .expect("a projection block implies a prior append installed the inverse");
                    for (i, row) in self.row_inv_sq.iter_mut().enumerate() {
                        *row = *row + row_sq_norm(&x, i);
                    }
                    for i in 0..s {
                        self.row_inv_sq.push(row_sq_norm(&g22, i));
                    }
                    // Every operand comes from this backend's kernels, so
                    // the memory orders agree by construction and the
                    // `replace_slice` order assertions cannot fire.
                    let mut g_new =
                        DenseTensorData::zeros_in_order(vec![p_new, p_new], g.data().order());
                    g_new.replace_slice(g.data(), &[0, 0]);
                    g_new.replace_slice(x.data(), &[0, p_old]);
                    g_new.replace_slice(g22.data(), &[p_old, p_old]);
                    self.g = Some(DenseTensor::from_data(g_new));
                }
            }
        }

        Ok(QrAppendOutcome::FullRank)
    }
}

/// Squared Euclidean norm of row `i` of a matrix, read through the
/// order-aware accessor.
fn row_sq_norm<T: Scalar>(m: &DenseTensor<T>, i: usize) -> T::Real {
    let mut acc = T::Real::zero();
    for j in 0..m.shape()[1] {
        let x = m.get([i, j]).abs();
        acc = acc + x * x;
    }
    acc
}
