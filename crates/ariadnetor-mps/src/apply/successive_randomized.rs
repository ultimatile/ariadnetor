//! Successive randomized compression (SRC): apply an MPO to an MPS — or a
//! coefficient-weighted sum of MPO-MPS products — via a single
//! right-to-left randomized-QB sweep, without materializing any
//! `w_R * chi_R` product MPS (algorithm from arXiv:2504.06475).
//!
//! Each output site is the Q factor of a randomized QB decomposition of the
//! partially compressed product. The test matrix is a Khatri-Rao product of
//! per-site Gaussian matrices; it is never formed explicitly. Instead, the
//! sketch columns carry a left-to-right recursion of small environment
//! tensors, computed once and reused across the sweep, so exactness with
//! probability one at the representable rank is preserved even though the
//! same Gaussians serve every site. Columns are created and recursed in
//! blocks — one batched contraction per site instead of per-column
//! tensordot chains — and the per-site adaptive loop feeds the resulting
//! panel blocks to an [`IncrementalQr`], so a growth round costs a
//! column-block update rather than a full restack, refactorization, and
//! inversion of everything sketched so far.
//!
//! For a linear combination `sum_t coeff_t * H_t psi_t` the sweep keeps one
//! environment store and one cap per term but shares each site's Gaussian
//! block across the terms' recursions. Sharing is a correctness
//! requirement, not an optimization: the test matrix lives on the output
//! physical legs, which the terms share, so the coefficient-weighted sum of
//! per-term panels is exactly the sketch of the summed operand — with
//! independent per-term draws a cancelling sum would sketch as nonzero.
//! Coefficients enter only where per-term tensors are summed — the sketch
//! panels and the deterministic site assembly (the first site, or the
//! whole chain on the single-site path); a cap is a projection of one
//! term's remaining network through the shared Q, so scaling it would
//! reapply the coefficient at every later site. All certification
//! machinery (QR, norm accumulator, stopping rule) sees only the summed
//! quantities and is unchanged from the single-pair sweep.
//!
//! Dense-only: a Gaussian sketch mixes symmetry sectors, so there is no
//! block-sparse twin (the dispatch arm panics instead).
//!
//! Non-finite poisoning (an overflowed contraction producing `inf`, or
//! `inf - inf` / `0 * inf` producing NaN) is surfaced as
//! [`ApplyError::NonFinite`] at the result boundary instead of flowing
//! on as a success: every growth round scans its (summed) sketch panel
//! for non-finite elements, and every assembled site tensor is scanned
//! before the state is returned (see
//! [`apply_sum_successive_randomized_dense`]). The adaptive stopping rule
//! is additionally hardened against representational extremes that carry
//! finite elements but degenerate the certification quantities: the
//! accumulated sketch norm is kept in a saturation-free scaled form, an
//! overflowed error estimator counts as "not converged" rather than
//! certifying, and a non-finite QR diagonal (a panel column norm past
//! the real type's range) surfaces as [`ApplyError::NonFinite`] instead
//! of masquerading as rank deficiency.
//!
//! No code is ported from the paper's reference implementation
//! (RandomMPOMPS, <https://github.com/chriscamano/RandomMPOMPS>); this
//! module is written from the published algorithm against this crate's
//! primitives. Behavioral choices that the paper leaves open — real
//! Gaussian sketches for complex scalars, clamping rather than rejecting
//! out-of-range dimensions, and the linear-combination structure (shared
//! per-site Gaussian blocks, per-term environments and caps, coefficients
//! applied at sketch accumulation and first-site assembly only) — follow
//! that implementation and its matured successor
//! (TNrandNLA, <https://github.com/chriscamano/TNrandNLA>), and "the
//! reference implementation" in comments below refers to them.

use ariadnetor_core::{NormAccumulator, Scalar, scale_safe_norm};
use ariadnetor_linalg::{IncrementalQr, QrAppendOutcome, permute_with_backend, tensordot};
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensor, OpsFor};
use num_traits::{Float, NumCast, Zero};
use rand::SeedableRng;
use rand::rngs::StdRng;

mod env;
mod terms;
#[cfg(test)]
mod tests;

use env::{EnvStore, extend_new_columns, sketch_panel_block};
pub(crate) use terms::DenseTerm;
use terms::{prune_zero_terms, validate_terms, weighted_sum};

use super::super::chain::TensorChain;
use super::super::types::{
    ApplyError, CanonicalForm, Mpo, Mps, SuccessiveRandomizedParams, TruncateParams,
};
use super::check_finite;

/// Cast a plain-`f64` tolerance into `T::Real`, panicking when the value is
/// not representable as a finite real. `NumCast::from` maps an out-of-range
/// `f64` to `Some(inf)` for `f32` targets rather than `None`; without the
/// post-cast finiteness check, a comparison like `err <= cutoff * norm`
/// would always hold and report spurious convergence.
fn real_from_f64<T: Scalar>(x: f64) -> T::Real {
    ariadnetor_core::try_real_from_f64::<T>(x)
        .expect("cutoff must be representable as a finite value in the scalar's real type")
}

/// Leave-one-out error estimate from the row norms of `R^-1` maintained by
/// the incremental QR: `sqrt((1/p) * sum_i ||g_i||^-2)` over the columns
/// `g_i` of `G = R^-dagger` (arXiv:2504.06475) — row norms of `R^-1` are
/// the same quantities. The reciprocal sum is accumulated with the
/// scale-safe kernel so the squares of the reciprocals are never formed:
/// the row norms scale as the inverse of the sketched state's amplitude,
/// and squaring either them or their reciprocals leaves the representable
/// range long before the estimate itself does. Rank deficiency is
/// reported by the QR append itself ([`QrAppendOutcome::RankDeficient`])
/// before this runs, so the row norms fed here always come from an
/// invertible factor and are nonzero (every row of `R^-1` carries the
/// nonzero diagonal `1 / R_ii`).
///
/// Returns `None` when the estimate carries no information: a non-finite
/// row norm means the maintained inverse overflowed (indistinguishable,
/// at this point, from a genuinely huge row norm whose true contribution
/// would be negligible), and a non-finite finished estimate means the
/// reciprocal accumulation itself saturated — feeding either into the
/// stopping comparison could report convergence without the true
/// ordering holding (`inf <= inf`, or an overflow-collapsed zero passing
/// any cutoff). `Some(err)` implies `err` is finite.
fn leave_one_out_estimate<T: Scalar>(row_norms: &[T::Real]) -> Option<T::Real> {
    if row_norms.iter().any(|r| !r.is_finite()) {
        return None;
    }
    let p = row_norms.len();
    debug_assert!(
        row_norms.iter().all(|r| *r > T::Real::zero()),
        "row norms of an invertible R^-1 are positive"
    );
    let inv_norm = scale_safe_norm(row_norms.iter().map(|r| r.recip()));
    let p_real = <T::Real as NumCast>::from(p).expect("bond dimensions fit in the real type");
    let err = inv_norm / p_real.sqrt();
    err.is_finite().then_some(err)
}

/// Validate the SRC parameter struct. Every field is checked in every mode
/// (configuration hygiene) even though fixed mode consults only
/// `output_dim` / `max_dim` / `seed`.
fn validate(src: &SuccessiveRandomizedParams) {
    assert!(src.sketch_dim >= 1, "sketch_dim must be at least 1");
    assert!(
        src.sketch_increment >= 1,
        "sketch_increment must be at least 1"
    );
    assert!(src.min_dim >= 1, "min_dim must be at least 1");
    assert_ne!(
        src.output_dim,
        Some(0),
        "output_dim must be at least 1 when fixed"
    );
    assert_ne!(src.max_dim, Some(0), "max_dim must be at least 1 when set");
    if let Some(c) = src.cutoff {
        assert!(
            c.is_finite() && c >= 0.0,
            "cutoff must be finite and non-negative"
        );
    }
    assert!(
        src.output_dim.is_some() || src.cutoff.is_some(),
        "adaptive mode (output_dim = None) requires a cutoff"
    );
}

/// Apply a Dense MPO to a Dense MPS via successive randomized compression.
///
/// Thin wrapper over [`apply_sum_successive_randomized_dense`] with a
/// single coefficient-one term; the sum kernel's coefficient-one bypass
/// keeps the arithmetic — and therefore the output — bit-identical to a
/// dedicated single-pair sweep at equal seeds.
///
/// See [`super::super::types::ApplyMethod::SuccessiveRandomized`] and
/// [`SuccessiveRandomizedParams`] for semantics and panics.
///
/// # Errors
///
/// See [`apply_sum_successive_randomized_dense`].
pub(crate) fn apply_successive_randomized_dense<T, B>(
    backend: &B,
    op: &Mpo<DenseStorage<T>, DenseLayout>,
    psi: &Mps<DenseStorage<T>, DenseLayout>,
    params: Option<&TruncateParams>,
    src: SuccessiveRandomizedParams,
) -> Result<Mps<DenseStorage<T>, DenseLayout>, ApplyError>
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    apply_sum_successive_randomized_dense(backend, &[(op, psi)], &[T::one()], params, src)
}

/// Apply a coefficient-weighted sum of Dense MPO-MPS products via
/// successive randomized compression: `eta ~ sum_t coeffs[t] * H_t psi_t`
/// in one sweep, sharing each site's Gaussian block across terms (see the
/// module doc for why sharing is a correctness requirement).
///
/// Single right-to-left sweep; sites `1..n` of the result are
/// right-orthogonal by construction (each is a thin-QR Q factor over its
/// physical and right legs), so the result is `Mixed { center: 0 }`. When
/// `params` is `Some`, the standard `canonicalize` + `truncate` finishing
/// pass runs afterwards, matching the streaming-naive convention.
///
/// The default per-bond cap (`max_dim = None`) is the sum over terms of
/// the products of maximum MPO and MPS bond dimensions — the rank bound
/// rank subadditivity puts on the summed product — in both stopping modes.
///
/// Zero-weighted terms are validated but otherwise behave as absent: they
/// are pruned before the sweep (see [`prune_zero_terms`]), and an
/// all-zero coefficient list yields the bond-dimension-1 zero state.
///
/// # Panics
///
/// Panics on an empty term list, zero-length chains, mismatched chain
/// lengths, a coefficient count differing from the term count, non-finite
/// coefficients, a within-term MPO-ket / MPS-physical dimension mismatch,
/// or MPO output dimensions differing across terms — and on the parameter
/// violations documented in [`SuccessiveRandomizedParams`].
///
/// # Errors
///
/// Returns [`ApplyError::NonFinite`] when a non-finite element (NaN/inf)
/// reaches a result boundary — a growth round's summed sketch panel or an
/// assembled site tensor — instead of letting the poisoned state flow on
/// as a success. Every returned site tensor is scanned, so an `Ok` state
/// contains only finite elements before the optional finishing pass.
/// Non-finite values arising only inside that finishing pass
/// (`canonicalize` + `truncate`) are the truncation machinery's concern
/// and are not checked here. The element detector itself does not reject
/// a finite state whose Frobenius norm merely overflows `T::Real`, but
/// adaptive mode additionally errors when a sketch panel's column norm
/// overflow degenerates the QR factorization
/// ([`QrAppendOutcome::NonFinite`]) — the stopping rule cannot certify
/// anything against a factor that does not exist, and stopping there
/// would silently return a bond-stuck state. Fixed mode never certifies,
/// so it tolerates the degenerated factor and relies on the elementwise
/// scans alone.
pub(crate) fn apply_sum_successive_randomized_dense<T, B>(
    backend: &B,
    terms: &[DenseTerm<'_, T>],
    coeffs: &[T],
    params: Option<&TruncateParams>,
    src: SuccessiveRandomizedParams,
) -> Result<Mps<DenseStorage<T>, DenseLayout>, ApplyError>
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    let n = validate_terms(terms, coeffs);
    validate(&src);
    // Converted up front so an unrepresentable cutoff fails fast in every
    // mode, not only on the adaptive path.
    let cutoff_real: Option<T::Real> = src.cutoff.map(real_from_f64::<T>);

    // Zero-weighted terms behave as absent (see `prune_zero_terms`). When
    // everything is pruned the sum is the zero state, represented at bond
    // dimension 1 over the validated output physical dimensions. Only the
    // center site is a zero tensor; the off-center sites are normalized
    // basis tensors, which are genuine right-isometries — the claimed
    // `Mixed { center: 0 }` form must hold structurally, because
    // downstream consumers (`truncate` among them) trust the metadata and
    // skip re-canonicalization.
    let (terms_kept, coeffs_kept) = prune_zero_terms(terms, coeffs);
    if terms_kept.is_empty() {
        let sites = (0..n)
            .map(|i| {
                let d = terms[0].0.site(i).shape()[2];
                let mut site = DenseTensor::<T>::zeros(vec![1, d, 1]);
                if i > 0 {
                    site.set([0, 0, 0], num_traits::One::one());
                }
                site
            })
            .collect();
        let mut result: Mps<DenseStorage<T>, DenseLayout> = Mps::from_sites(sites);
        result.set_canonical_form(CanonicalForm::Mixed { center: 0 });
        super::finish_dense(backend, &mut result, params);
        return Ok(result);
    }
    let (terms, coeffs) = (&terms_kept[..], &coeffs_kept[..]);

    if n == 1 {
        // No bond to compress: the exact weighted sum of local products is
        // the answer.
        let locals: Vec<DenseTensor<T>> = terms
            .iter()
            .map(|(op, psi)| {
                tensordot(backend, op.site(0), psi.site(0), &[0, 1], &[0, 1])
                    .expect("single-site product: MPO/MPS site legs must be compatible")
            })
            .collect();
        // (b, w_r = 1, chi_r = 1) -> (1, d, 1).
        let d = terms[0].0.site(0).shape()[2];
        let site = weighted_sum(locals, coeffs).reshape_logical(vec![1, d, 1]);
        // No sketch exists on this path, so the exact product itself is
        // the boundary quantity to check.
        check_finite(0, &site)?;
        let mut result: Mps<DenseStorage<T>, DenseLayout> = Mps::from_sites(vec![site]);
        result.set_canonical_form(CanonicalForm::Mixed { center: 0 });
        super::finish_dense(backend, &mut result, params);
        return Ok(result);
    }

    let mut rng = StdRng::seed_from_u64(src.seed);
    let maxdim_global = src.max_dim.unwrap_or_else(|| {
        // Saturating: an overflowed cap must pin at "effectively
        // unbounded" rather than wrap to a small bound that would silently
        // truncate ranks.
        terms
            .iter()
            .map(|(op, psi)| op.max_bond_dim().saturating_mul(psi.max_bond_dim()))
            .fold(0usize, usize::saturating_add)
    });
    // Adaptive mode reads the estimator off the incremental QR; fixed mode
    // does a single append and never consults it, so it skips the tracking.
    let adaptive = src.output_dim.is_none();

    let mut envs: Vec<EnvStore<T>> = (0..terms.len()).map(|_| EnvStore::new(n)).collect();
    // Right boundary caps: dummy (c = 1, w' = 1, chi' = 1) per term so the
    // last site goes through the same code path as the interior ones.
    let mut caps: Vec<DenseTensor<T>> = (0..terms.len())
        .map(|_| DenseTensor::<T>::ones(vec![1, 1, 1]))
        .collect();
    // Sites n-1 down to 1, collected right to left.
    let mut rev_sites: Vec<DenseTensor<T>> = Vec::with_capacity(n - 1);

    for j in (1..n).rev() {
        // Output dimension and cap rank are shared across terms, so every
        // term's panel has the same `rows` fused row count.
        let d = terms[0].0.site(j).shape()[2];
        let cap_dim = caps[0].shape()[0];
        let rows = d * cap_dim;
        // The compressed product's bond at this cut cannot exceed the sum
        // over terms of the left bond products (rank subadditivity over
        // each term's exactly representable rank), nor the QR row count.
        // Left legs, not `bond_dim(j)` (the right bond), bound the rank of
        // the matrix this site's QB step sees.
        // Saturating for the same reason as the `max_dim` default: a
        // wrapped sum would masquerade as a small rank cap.
        let left_rank_sum: usize = terms
            .iter()
            .map(|(op, psi)| op.site(j).shape()[0].saturating_mul(psi.site(j).shape()[0]))
            .fold(0usize, usize::saturating_add);
        let current_maxdim = left_rank_sum.min(maxdim_global).min(rows);
        let current_mindim = src.min_dim.min(current_maxdim);
        let mut target_p = match src.output_dim {
            Some(k) => k.min(current_maxdim),
            None => src.sketch_dim.min(current_maxdim).max(current_mindim),
        };

        let mut inc = IncrementalQr::<T>::new(rows, adaptive);
        // Accumulated Frobenius norm of the summed sketch, kept in the
        // accumulator's `(scale, sumsq)` form instead of a finished
        // scalar: panels are elementwise finite (enforced below), so the
        // scale stays finite and the representation never saturates —
        // where a finished running norm would overflow to `inf` once the
        // true accumulated norm left the representable range and make
        // `err <= cutoff * inf` accept any finite estimate at the first
        // round. Elements enter through the component-wise
        // [`NormAccumulator::push_scalar`], which owns the reason the
        // modulus must not be formed first.
        let mut norm_acc = NormAccumulator::new();
        loop {
            let columns_created = envs[0].ncols();
            if target_p > columns_created {
                extend_new_columns(
                    backend,
                    terms,
                    &mut envs,
                    j - 1,
                    target_p - columns_created,
                    &mut rng,
                );
            }
            let c0 = inc.ncols();
            let panels: Vec<DenseTensor<T>> = terms
                .iter()
                .enumerate()
                .map(|(t, (op, psi))| {
                    let env_block = DenseTensor::from_data(envs[t].range(j - 1, c0, target_p));
                    sketch_panel_block(backend, &env_block, op.site(j), psi.site(j), &caps[t])
                })
                .collect();
            let panel = weighted_sum(panels, coeffs);
            // Poison detector: any non-finite value that entered this
            // round's summed sketch is caught here, in both adaptive and
            // fixed mode. Scanning per round (rather than only the
            // assembled result) fails on the round the poison appears
            // instead of growing the sketch to its bound first.
            check_finite(j, &panel)?;
            // Fixed mode breaks before either consumer of the accumulated
            // norm (the zero-norm break and the estimator), so the panel
            // pass is adaptive-only work.
            if adaptive {
                for x in panel.data_slice() {
                    norm_acc.push_scalar(*x);
                }
            }
            // The `current_maxdim` clamp above keeps the block within the
            // factorization's bounds, so a failure here is an unrecoverable
            // backend error, not a rejected input.
            let outcome = inc.append(backend, &panel).expect(
                "sketch QR append: the clamps keep the block within the row budget, \
                     so only an unrecoverable backend kernel failure lands here",
            );
            // A non-finite QR diagonal (a panel column whose true norm
            // exceeds the real type's range) degenerates the
            // certification machinery itself: the rank test and the
            // estimator both read the factor this append could not
            // produce. In adaptive mode that must not escape through any
            // of the successful-coverage exits below (max-dimension,
            // zero-norm, rank-deficient), so it errors here; fixed mode
            // never claims certification and keeps relying on the
            // elementwise result-boundary scans instead.
            if adaptive && let QrAppendOutcome::NonFinite { diagnostic } = outcome {
                // The diagnostic is the degenerated diagonal magnitude,
                // reported from the detection site (non-finite by
                // construction).
                return Err(ApplyError::NonFinite {
                    site: j,
                    norm: diagnostic,
                });
            }
            let p = inc.ncols();

            if !adaptive || p == current_maxdim {
                break;
            }
            if norm_acc.scale().is_zero() {
                // Zero summed product at this cut (including exact
                // cancellation between terms): any orthonormal basis is
                // exact.
                break;
            }
            if outcome == QrAppendOutcome::RankDeficient {
                // Rank-deficient sketch = the exact rank is already covered.
                break;
            }
            let row_norms = inc
                .r_inverse_row_norms()
                .expect("adaptive mode tracks the inverse and this append was full-rank");
            let p_real =
                <T::Real as NumCast>::from(p).expect("bond dimensions fit in the real type");
            let cutoff = cutoff_real.expect("validated: adaptive mode has a cutoff");
            // p started at or above the clamped min_dim and only grows, so
            // the cutoff test alone decides convergence. The comparison is
            // `err <= cutoff * norm_est` with `norm_est = norm / sqrt(p)`,
            // evaluated in the log domain over the accumulator's
            // `(scale, sumsq)` representation: no product of these
            // factors is ever formed, so no ordering of finite operands
            // can overflow or underflow the comparison — a guarantee no
            // association order of the direct product gives, because a
            // representable-range cutoff times an extreme scale can
            // saturate an intermediate in either direction. Every log
            // here is finite: `err` is positive and finite by the
            // estimator's `Some` contract, `scale` by the zero-product
            // break above, `sumsq >= 1` once anything nonzero was
            // pushed, and `p >= 1` — except a zero `cutoff`, whose
            // `ln = -inf` makes the threshold unsatisfiable, so exact
            // compression is decided by the rank-deficiency break
            // alone (the pre-log behavior). The logs shift the decision
            // boundary only at ulp level, far inside the estimator's
            // stochastic spread. An uninformative `None` estimate stays
            // "not converged"; growth is bounded by the
            // `current_maxdim` break above.
            let half = <T::Real as NumCast>::from(0.5).expect("0.5 is representable");
            let converged = match leave_one_out_estimate::<T>(row_norms) {
                None => false,
                Some(err) => {
                    err.ln()
                        <= cutoff.ln()
                            + norm_acc.scale().ln()
                            + (norm_acc.sumsq().ln() - p_real.ln()) * half
                }
            };
            if converged {
                break;
            }
            target_p = p.saturating_add(src.sketch_increment).min(current_maxdim);
        }

        // The terminal accessor re-orthonormalizes the assembled basis
        // whenever more than one block was appended (fixed mode's single
        // plain Householder QR is returned as is), so the site tensor is an
        // isometry regardless of how block Gram-Schmidt fared across the
        // growth rounds. The cost is at most one O(rows p^2) factorization
        // per site — the same as a single full QR over the accumulated
        // sketch. One reachable exception: after a NonFinite append —
        // fixed mode only, since adaptive mode errored above — the
        // degenerated factorization voids the accessor's isometry claim,
        // and the result-boundary scans below bound the damage to finite
        // values without restoring orthonormality (see the fixed-mode
        // tolerance note at the append site).
        let q = inc
            .into_orthonormal_q(backend)
            .expect("terminal re-orthonormalization: Q is a valid matrix");

        // Q columns are orthonormal over the fused (d, cap_dim) rows, so the
        // permuted site is an isometry from its left bond into (physical x
        // right bond): right-orthogonal by construction.
        let k = q.shape()[1];
        let q_site = q.split_leg(0, &[d, cap_dim]);
        let site = permute_with_backend(backend, &q_site, &[2, 0, 1])
            .expect("site permute: rank-3 permutation is always well-formed");

        // Cap updates: absorb the conjugated new site into each term's
        // running right environment, (c_new = k, w_t, chi_t) for the next
        // site. The shared Q keeps the leading rank `k` common to all
        // terms.
        for (t, (op, psi)) in terms.iter().enumerate() {
            let t1 = tensordot(backend, &site.conj(), &caps[t], &[2], &[0])
                .expect("cap contraction: MPO/MPS site legs must be compatible");
            let t2 = tensordot(backend, &t1, op.site(j), &[1, 2], &[2, 3])
                .expect("cap contraction: MPO/MPS site legs must be compatible");
            caps[t] = tensordot(backend, &t2, psi.site(j), &[1, 3], &[2, 1])
                .expect("cap contraction: MPO/MPS site legs must be compatible");
            debug_assert_eq!(caps[t].shape()[0], k);
            // This iteration was `bufs[j - 1]`'s last reader (site j - 1
            // reads `bufs[j - 2]`), so its environments can be freed now
            // instead of holding every site's buffer until the sweep ends.
            envs[t].release(j - 1);
        }
        rev_sites.push(site);
    }

    // First site: deterministic weighted sum of per-term contractions with
    // the final caps; it carries whatever weight the orthonormal sites
    // cannot, i.e. the center.
    let mats: Vec<DenseTensor<T>> = terms
        .iter()
        .enumerate()
        .map(|(t, (op, psi))| {
            let t1 = tensordot(backend, op.site(0), psi.site(0), &[0, 1], &[0, 1])
                .expect("first-site contraction: MPO/MPS site legs must be compatible");
            tensordot(backend, &t1, &caps[t], &[1, 2], &[1, 2])
                .expect("first-site contraction: MPO/MPS site legs must be compatible")
        })
        .collect();
    let d0 = terms[0].0.site(0).shape()[2];
    let c0 = caps[0].shape()[0];
    let site0 = weighted_sum(mats, coeffs).reshape_logical(vec![1, d0, c0]);

    let mut sites = Vec::with_capacity(n);
    sites.push(site0);
    sites.extend(rev_sites.into_iter().rev());
    // Single enforcement point for the returned-state invariant: every
    // assembled site is scanned, so poison arising past the last panel
    // check — inside the QR factorization, the final cap updates, or the
    // first-site contraction — cannot reach an `Ok`. A new emission path
    // added to the sweep is covered here without needing its own check.
    for (idx, s) in sites.iter().enumerate() {
        check_finite(idx, s)?;
    }
    let mut result: Mps<DenseStorage<T>, DenseLayout> = Mps::from_sites(sites);
    result.set_canonical_form(CanonicalForm::Mixed { center: 0 });
    super::finish_dense(backend, &mut result, params);
    Ok(result)
}
