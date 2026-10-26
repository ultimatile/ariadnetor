//! Two-site gate application on a Dense MPS: contract a neighboring pair,
//! apply the operator, and split the result back by truncated SVD.

use ariadnetor_core::Scalar;
use ariadnetor_linalg::{TruncSvdParams, contract, diagonal_scale, trunc_svd};
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensor, OpsFor};

use super::canonicalize::{left_qr_step, right_lq_step};
use super::chain::TensorChain;
use super::types::{CanonicalForm, Mps, TruncResult};

/// Which site of the gated pair carries the orthogonality center after
/// [`Mps::apply_two_site_gate`] splits it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateCenter {
    /// The center lands on `site`; `site + 1` is right-isometric.
    Left,
    /// The center lands on `site + 1`; `site` is left-isometric.
    Right,
}

impl<T: Scalar> Mps<DenseStorage<T>, DenseLayout> {
    /// Apply the two-site operator `gate` to sites `site` and `site + 1`,
    /// truncating the new bond between them according to `svd`.
    ///
    /// `gate` has legs `(out_site, out_site+1, in_site, in_site+1)`: it is
    /// the `reshape_logical([d, d', d, d'])` of a `d·d' × d·d'` operator
    /// matrix whose rows are outputs and whose columns are inputs, the
    /// orientation of the [`SiteOps`](crate::SiteOps) matrices. It need not
    /// be unitary.
    ///
    /// The sites outside the pair must be isometric toward it. Whatever part
    /// of that the canonical form does not already guarantee is established
    /// first by QR / LQ steps that move the gauge toward the pair, so a center
    /// `k` sites away costs `k` steps and a center on the pair costs none
    /// (`Unknown` canonicalizes the whole chain). On return the form is
    /// `Mixed` with the center on the site `center` selects.
    ///
    /// Returns the Frobenius norm of the discarded singular values. The
    /// sites the canonical form covers are trusted to be isometric as it
    /// claims, not re-checked; when they are, this equals the absolute state
    /// error the truncation introduced. The state is not
    /// renormalized.
    ///
    /// # Panics
    ///
    /// Panics if `site + 1 >= self.len()`, if `gate.shape()` is not
    /// `[d_site, d_site+1, d_site, d_site+1]` for the physical dimensions of
    /// the two sites, if [`TruncSvdParams::validate`] rejects `svd`, or if the stored
    /// canonical form is inconsistent with the chain length `n` (a `Mixed`
    /// center at or past `n`, a `Partial` `right_start` past `n`, or a
    /// `Partial` whose `left_end` exceeds its `right_start`). Also panics if the chain is
    /// malformed (a site that is not rank 3, or adjacent bond dimensions
    /// that disagree) or if an internal linear-algebra call returns an error.
    pub fn apply_two_site_gate<B: OpsFor<DenseStorage<T>>>(
        &mut self,
        backend: &B,
        site: usize,
        gate: &DenseTensor<T>,
        svd: &TruncSvdParams,
        center: GateCenter,
    ) -> TruncResult<T> {
        let n = self.len();
        assert!(
            site < n.saturating_sub(1),
            "apply_two_site_gate: site {site} has no right neighbor in a chain of length {n}",
        );
        let d_l = self.site(site).shape()[1];
        let d_r = self.site(site + 1).shape()[1];
        assert_eq!(
            gate.shape(),
            [d_l, d_r, d_l, d_r],
            "apply_two_site_gate: gate shape must be [d_site, d_site+1, d_site, d_site+1]",
        );
        if let Err(err) = svd.validate() {
            panic!("apply_two_site_gate: invalid truncation parameters: {err}");
        }

        // The error identity needs sites `0..site` left-isometric and
        // `site + 2..n` right-isometric. Extend whichever side the form does
        // not already guarantee, one state-preserving step at a time, so the
        // cost scales with the distance to the pair rather than with `n`.
        let (left, right) = self.canonical_form().isometric_bounds(n);
        assert!(
            left <= right && right <= n,
            "apply_two_site_gate: canonical form {:?} is inconsistent with a chain of length {n}",
            self.canonical_form(),
        );
        for j in left..site {
            left_qr_step(self, j, backend);
        }
        for j in (site + 2..right).rev() {
            right_lq_step(self, j, backend);
        }

        let theta = contract(
            backend,
            self.site(site),
            self.site(site + 1),
            "lar,rbs->labs",
        )
        .expect("pair contraction: adjacent sites of a well-formed chain share their bond");
        let theta = contract(backend, gate, &theta, "xyab,labr->lxyr")
            .expect("gate contraction: validated by entry point");
        let chi_l = theta.shape()[0];
        let chi_r = theta.shape()[3];

        let (u, s, vt, error) =
            trunc_svd(backend, &theta, 2, svd).expect("trunc_svd failed during gate application");
        let u = u.split_leg(0, &[chi_l, d_l]);
        let vt = vt.split_leg(1, &[d_r, chi_r]);
        let (left, right, center_site) = match center {
            GateCenter::Left => {
                let us = diagonal_scale(backend, &u, s.data_slice(), 2)
                    .expect("U·S scaling failed during gate application");
                (us, vt, site)
            }
            GateCenter::Right => {
                let svt = diagonal_scale(backend, &vt, s.data_slice(), 0)
                    .expect("S·Vt scaling failed during gate application");
                (u, svt, site + 1)
            }
        };

        // `site_mut` resets the form to `Unknown`, so set it after the writes.
        *self.site_mut(site) = left;
        *self.site_mut(site + 1) = right;
        self.set_canonical_form(CanonicalForm::Mixed {
            center: center_site,
        });
        TruncResult { error }
    }
}
