//! Successive randomized compression (SRC): apply an MPO to an MPS via a
//! single right-to-left randomized-QB sweep, without materializing the
//! `w_R * chi_R` product MPS (algorithm from arXiv:2504.06475).
//!
//! Each output site is the Q factor of a randomized QB decomposition of the
//! partially compressed product. The test matrix is a Khatri-Rao product of
//! per-site Gaussian matrices; it is never formed explicitly. Instead, each
//! sketch column carries a left-to-right recursion of small environment
//! tensors, computed once and reused across the sweep, so exactness with
//! probability one at the representable rank is preserved even though the
//! same Gaussians serve every site.
//!
//! Dense-only: a Gaussian sketch mixes symmetry sectors, so there is no
//! block-sparse twin (the dispatch arm panics instead).
//!
//! No code is ported from the paper's reference implementation
//! (RandomMPOMPS, <https://github.com/chriscamano/RandomMPOMPS>); this
//! module is written from the published algorithm against this crate's
//! primitives. Behavioral choices that the paper leaves open — real
//! Gaussian sketches for complex scalars, clamping rather than rejecting
//! out-of-range dimensions — follow that implementation, and "the
//! reference implementation" in comments below refers to it.

use ariadnetor_core::Scalar;
use ariadnetor_linalg::{inverse_with_backend, permute_with_backend, qr, tensordot};
use ariadnetor_tensor::{DenseLayout, DenseStorage, DenseTensor, DenseTensorData, OpsFor};
use num_traits::{Float, NumCast, Zero};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;

use super::super::chain::TensorChain;
use super::super::types::{CanonicalForm, Mpo, Mps, SuccessiveRandomizedParams, TruncateParams};

/// Cast a plain-`f64` tolerance into `T::Real`, panicking when the value is
/// not representable as a finite real. `NumCast::from` maps an out-of-range
/// `f64` to `Some(inf)` for `f32` targets rather than `None`; without the
/// post-cast finiteness check, a comparison like `err <= cutoff * norm`
/// would always hold and report spurious convergence.
fn real_from_f64<T: Scalar>(x: f64) -> T::Real {
    <T::Real as NumCast>::from(x)
        .filter(|c| c.is_finite())
        .expect("cutoff must be representable as a finite value in the scalar's real type")
}

/// A rank-1 tensor of `len` i.i.d. standard Gaussians. Entries are real
/// even for complex scalars: the leave-one-out estimator only needs sketch
/// columns satisfying the isotropy condition `E[w w^dagger] = I`, which
/// real standard normals provide for complex operands too (the reference
/// implementation makes the same choice; see the module doc).
fn gaussian_vec<T: Scalar, R: RngExt>(len: usize, rng: &mut R) -> DenseTensor<T> {
    let mut v = DenseTensor::<T>::zeros(vec![len]);
    for i in 0..len {
        let x: f64 = rng.sample(StandardNormal);
        let re = <T::Real as NumCast>::from(x)
            .expect("a standard normal sample is representable in every supported real type");
        v.set([i], T::from_real_imag(re, T::Real::zero()));
    }
    v
}

/// One sketch column: the prefix chain of environment tensors. Each
/// Gaussian vector is drawn on the fly and fully absorbed into its site's
/// environment update, so only the environments persist. `envs[i]` has
/// shape `(w, chi)` — the MPO / MPS bonds to the right of site `i` — and
/// absorbs sites `0..=i`. Site `j` of the sweep consumes `envs[j - 1]`;
/// because the sweep moves right to left, a prefix computed once serves
/// every later (more-left) site.
struct SketchColumn<T: Scalar> {
    envs: Vec<DenseTensor<T>>,
}

impl<T: Scalar> SketchColumn<T> {
    fn new() -> Self {
        Self { envs: Vec::new() }
    }

    /// Extend this column's environment chain through site `last` (i.e.
    /// ensure `envs[0..=last]` exist), drawing Gaussians as needed.
    fn extend_through<B: OpsFor<DenseStorage<T>>>(
        &mut self,
        backend: &B,
        op: &Mpo<DenseStorage<T>, DenseLayout>,
        psi: &Mps<DenseStorage<T>, DenseLayout>,
        last: usize,
        rng: &mut StdRng,
    ) {
        for i in self.envs.len()..=last {
            let w = op.site(i);
            let a = psi.site(i);
            let omega = gaussian_vec::<T, _>(w.shape()[2], rng);
            // (b) x (w, k, b, w') over the bra leg -> (w, k, w').
            let m = tensordot(backend, &omega, w, &[0], &[2])
                .expect("sketch contraction: MPO/MPS site legs must be compatible");
            let boundary;
            let prev = if i == 0 {
                boundary = DenseTensor::<T>::ones(vec![1, 1]);
                &boundary
            } else {
                &self.envs[i - 1]
            };
            // (w, chi) x (w, k, w') -> (chi, k, w'), then contract the MPS
            // site over (chi, k) -> (w', chi').
            let t = tensordot(backend, prev, &m, &[0], &[0])
                .expect("sketch contraction: MPO/MPS site legs must be compatible");
            let env = tensordot(backend, &t, a, &[0, 1], &[0, 1])
                .expect("sketch contraction: MPO/MPS site legs must be compatible");
            self.envs.push(env);
        }
    }
}

/// Per-column sketch of site `j`: contract the column's environment with
/// the site pair and the cap, yielding a `(d_bra, cap_dim)` panel.
fn sketch_panel<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    env: &DenseTensor<T>,
    w: &DenseTensor<T>,
    a: &DenseTensor<T>,
    cap: &DenseTensor<T>,
) -> DenseTensor<T> {
    // (w, chi) x (w, k, b, w') -> (chi, k, b, w')
    let m = tensordot(backend, env, w, &[0], &[0])
        .expect("sketch contraction: MPO/MPS site legs must be compatible");
    // ... x (chi, k, chi') -> (b, w', chi')
    let t = tensordot(backend, &m, a, &[0, 1], &[0, 1])
        .expect("sketch contraction: MPO/MPS site legs must be compatible");
    // ... x (c, w', chi') -> (b, c)
    tensordot(backend, &t, cap, &[1, 2], &[1, 2])
        .expect("sketch contraction: MPO/MPS site legs must be compatible")
}

/// Leave-one-out error estimate from the square triangular factor `R`.
///
/// Returns `None` when `R` is numerically rank-deficient (some
/// `|R_ii| <= eps * p * max_j |R_jj|`): for a Gaussian sketch this happens
/// with probability one only when the sketch already covers the exact rank
/// of the compressed product, so the caller treats `None` as "converged"
/// rather than inverting a singular matrix.
///
/// Otherwise the estimate is `sqrt((1/p) * sum_i ||g_i||^-2)` over the
/// columns `g_i` of `G = R^{-dagger}` (arXiv:2504.06475), computed as the
/// row norms of `R^{-1}` — the same quantities, for one plain inversion
/// with no conjugate-transpose materialization. Elements are read through
/// the order-aware accessor, never through raw storage strides.
fn leave_one_out_estimate<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    r: &DenseTensor<T>,
) -> Option<T::Real> {
    let p = r.shape()[0];
    debug_assert_eq!(p, r.shape()[1], "R must be square for the estimator");

    let mut max_diag = T::Real::zero();
    for i in 0..p {
        let d = r.get([i, i]).abs();
        if d > max_diag {
            max_diag = d;
        }
    }
    let p_real = <T::Real as NumCast>::from(p).expect("bond dimensions fit in the real type");
    let tol = T::Real::epsilon() * p_real * max_diag;
    for i in 0..p {
        if r.get([i, i]).abs() <= tol {
            return None;
        }
    }

    let g = inverse_with_backend(backend, r, 1)
        .expect("R inversion: diagonal checked non-singular above");
    let mut inv_sq_sum = T::Real::zero();
    for i in 0..p {
        let mut row_sq = T::Real::zero();
        for col in 0..p {
            let x = g.get([i, col]).abs();
            row_sq = row_sq + x * x;
        }
        inv_sq_sum = inv_sq_sum + row_sq.recip();
    }
    Some((inv_sq_sum / p_real).sqrt())
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
/// Single right-to-left sweep; sites `1..n` of the result are
/// right-orthogonal by construction (each is a thin-QR Q factor over its
/// physical and right legs), so the result is `Mixed { center: 0 }`. When
/// `params` is `Some`, the standard `canonicalize` + `truncate` finishing
/// pass runs afterwards, matching the streaming-naive convention.
///
/// See [`super::super::types::ApplyMethod::SuccessiveRandomized`] and
/// [`SuccessiveRandomizedParams`] for semantics and panics.
pub(crate) fn apply_successive_randomized_dense<T, B>(
    backend: &B,
    op: &Mpo<DenseStorage<T>, DenseLayout>,
    psi: &Mps<DenseStorage<T>, DenseLayout>,
    params: Option<&TruncateParams>,
    src: SuccessiveRandomizedParams,
) -> Mps<DenseStorage<T>, DenseLayout>
where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    let n = psi.len();
    assert_eq!(n, op.len(), "MPO and MPS lengths must match");
    assert!(n > 0, "must have at least one site");
    validate(&src);
    // Converted up front so an unrepresentable cutoff fails fast in every
    // mode, not only on the adaptive path.
    let cutoff_real: Option<T::Real> = src.cutoff.map(real_from_f64::<T>);

    if n == 1 {
        // No bond to compress: the exact local product is the answer.
        let w = op.site(0);
        let a = psi.site(0);
        let t = tensordot(backend, w, a, &[0, 1], &[0, 1])
            .expect("single-site product: MPO/MPS site legs must be compatible");
        // (b, w_r = 1, chi_r = 1) -> (1, d, 1).
        let d = w.shape()[2];
        let site = t.reshape_logical(vec![1, d, 1]);
        let mut result: Mps<DenseStorage<T>, DenseLayout> = Mps::from_sites(vec![site]);
        result.set_canonical_form(CanonicalForm::Mixed { center: 0 });
        super::finish_dense(backend, &mut result, params);
        return result;
    }

    let mut rng = StdRng::seed_from_u64(src.seed);
    let maxdim_global = src
        .max_dim
        .unwrap_or_else(|| op.max_bond_dim() * psi.max_bond_dim());

    let mut columns: Vec<SketchColumn<T>> = Vec::new();
    // Right boundary cap: dummy (c = 1, w' = 1, chi' = 1) so the last site
    // goes through the same code path as the interior ones.
    let mut cap = DenseTensor::<T>::ones(vec![1, 1, 1]);
    // Sites n-1 down to 1, collected right to left.
    let mut rev_sites: Vec<DenseTensor<T>> = Vec::with_capacity(n - 1);

    for j in (1..n).rev() {
        let w = op.site(j);
        let a = psi.site(j);
        let d = w.shape()[2];
        let cap_dim = cap.shape()[0];
        let rows = d * cap_dim;
        // The compressed product's bond at this cut cannot exceed the left
        // bond product of the site pair (its exactly representable rank),
        // nor the QR row count. Left legs, not `bond_dim(j)` (the right
        // bond), bound the rank of the matrix this site's QB step sees.
        let current_maxdim = (w.shape()[0] * a.shape()[0]).min(maxdim_global).min(rows);
        let current_mindim = src.min_dim.min(current_maxdim);
        let mut p = match src.output_dim {
            Some(k) => k.min(current_maxdim),
            None => src.sketch_dim.min(current_maxdim).max(current_mindim),
        };

        let mut panels: Vec<DenseTensor<T>> = Vec::new();
        let q = loop {
            columns.resize_with(p.max(columns.len()), SketchColumn::new);
            for column in columns.iter_mut().take(p).skip(panels.len()) {
                column.extend_through(backend, op, psi, j - 1, &mut rng);
                panels.push(sketch_panel(backend, &column.envs[j - 1], w, a, &cap));
            }
            // Stack the (d, cap_dim) panels into Y (d, cap_dim, p). All
            // panels come from the same tensordot pipeline, so they share
            // one memory order, which `stack` requires.
            let panel_data: Vec<&DenseTensorData<T>> = panels.iter().map(|t| t.data()).collect();
            let y = DenseTensor::from_data(DenseTensorData::stack(&panel_data, 2));
            // p <= rows (so R is square, as the estimator requires) is
            // enforced by the `current_maxdim` clamp above, which every
            // assignment to p passes through.
            debug_assert!(p <= rows, "sketch dimension exceeds the QR row count");
            let (q, r) = qr(backend, &y, 2).expect("sketch QR: shape validated above");

            if src.output_dim.is_some() || p == current_maxdim {
                break q;
            }
            let y_norm = y.norm();
            if y_norm.is_zero() {
                // Zero product at this cut: any orthonormal basis is exact.
                break q;
            }
            match leave_one_out_estimate(backend, &r) {
                // Rank-deficient sketch = the exact rank is already covered.
                None => break q,
                Some(err) => {
                    let p_real = <T::Real as NumCast>::from(p)
                        .expect("bond dimensions fit in the real type");
                    let norm_est = y_norm / p_real.sqrt();
                    let cutoff = cutoff_real.expect("validated: adaptive mode has a cutoff");
                    // p started at or above the clamped min_dim and only
                    // grows, so the cutoff test alone decides convergence.
                    if err <= cutoff * norm_est {
                        break q;
                    }
                }
            }
            p = p.saturating_add(src.sketch_increment).min(current_maxdim);
        };

        // Q columns are orthonormal over the fused (d, cap_dim) rows, so the
        // permuted site is an isometry from its left bond into (physical x
        // right bond): right-orthogonal by construction.
        let k = q.shape()[1];
        let q_site = q.split_leg(0, &[d, cap_dim]);
        let site = permute_with_backend(backend, &q_site, &[2, 0, 1])
            .expect("site permute: rank-3 permutation is always well-formed");

        // Cap update: absorb the conjugated new site into the running
        // right environment, (c_new = k, w, chi) for the next site.
        let t1 = tensordot(backend, &site.conj(), &cap, &[2], &[0])
            .expect("cap contraction: MPO/MPS site legs must be compatible");
        let t2 = tensordot(backend, &t1, w, &[1, 2], &[2, 3])
            .expect("cap contraction: MPO/MPS site legs must be compatible");
        cap = tensordot(backend, &t2, a, &[1, 3], &[2, 1])
            .expect("cap contraction: MPO/MPS site legs must be compatible");
        debug_assert_eq!(cap.shape()[0], k);

        rev_sites.push(site);
    }

    // First site: deterministic contraction with the final cap; it carries
    // whatever weight the orthonormal sites cannot, i.e. the center.
    let w0 = op.site(0);
    let a0 = psi.site(0);
    let t1 = tensordot(backend, w0, a0, &[0, 1], &[0, 1])
        .expect("first-site contraction: MPO/MPS site legs must be compatible");
    let site0_mat = tensordot(backend, &t1, &cap, &[1, 2], &[1, 2])
        .expect("first-site contraction: MPO/MPS site legs must be compatible");
    let d0 = w0.shape()[2];
    let c0 = cap.shape()[0];
    let site0 = site0_mat.reshape_logical(vec![1, d0, c0]);

    let mut sites = Vec::with_capacity(n);
    sites.push(site0);
    sites.extend(rev_sites.into_iter().rev());
    let mut result: Mps<DenseStorage<T>, DenseLayout> = Mps::from_sites(sites);
    result.set_canonical_form(CanonicalForm::Mixed { center: 0 });
    super::finish_dense(backend, &mut result, params);
    result
}
