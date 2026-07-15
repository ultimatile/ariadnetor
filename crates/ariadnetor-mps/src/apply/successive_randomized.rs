//! Successive randomized compression (SRC): apply an MPO to an MPS via a
//! single right-to-left randomized-QB sweep, without materializing the
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
use ariadnetor_linalg::{
    IncrementalQr, QrAppendOutcome, einsum_with_backend, permute_with_backend, tensordot,
};
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

/// A `(cols, len)` block of i.i.d. standard Gaussians, one row per sketch
/// column. Entries are real even for complex scalars: the leave-one-out
/// estimator only needs sketch columns satisfying the isotropy condition
/// `E[w w^dagger] = I`, which real standard normals provide for complex
/// operands too (the reference implementation makes the same choice; see
/// the module doc). Each column's vector is drawn contiguously, so a
/// single-column block consumes the stream exactly like a lone vector.
fn gaussian_block<T: Scalar, R: RngExt>(cols: usize, len: usize, rng: &mut R) -> DenseTensor<T> {
    let mut m = DenseTensor::<T>::zeros(vec![cols, len]);
    for c in 0..cols {
        for i in 0..len {
            let x: f64 = rng.sample(StandardNormal);
            let re = <T::Real as NumCast>::from(x)
                .expect("a standard normal sample is representable in every supported real type");
            m.set([c, i], T::from_real_imag(re, T::Real::zero()));
        }
    }
    m
}

/// One site's batched environment buffer: physical shape
/// `(w, chi, capacity)` with the batch axis last, holding the environments
/// of the first `filled` sketch columns. Capacity grows geometrically so
/// appends cost amortized `O(block)`, not `O(buffer)`.
struct EnvBuf<T: Scalar> {
    data: DenseTensorData<T>,
    filled: usize,
}

impl<T: Scalar> EnvBuf<T> {
    /// Append a `(w, chi, s)` column block. The block's memory order sets
    /// the buffer's order at creation and must match on later appends
    /// (`replace_slice` enforces it); every block comes from the same
    /// einsum pipeline, so the orders agree by construction.
    fn append(&mut self, block: &DenseTensorData<T>) {
        let s = block.shape()[2];
        let (w, chi, cap) = (
            self.data.shape()[0],
            self.data.shape()[1],
            self.data.shape()[2],
        );
        if self.filled + s > cap {
            let new_cap = (cap * 2).max(self.filled + s);
            let mut grown =
                DenseTensorData::zeros_in_order(vec![w, chi, new_cap], self.data.order());
            let prefix = self.data.slice(&[(0, w), (0, chi), (0, self.filled)]);
            grown.replace_slice(&prefix, &[0, 0, 0]);
            self.data = grown;
        }
        self.data.replace_slice(block, &[0, 0, self.filled]);
        self.filled += s;
    }

    /// Copy out the environments of columns `c0..c1` as a `(w, chi, c1-c0)`
    /// tensor.
    fn range(&self, c0: usize, c1: usize) -> DenseTensorData<T> {
        assert!(c0 < c1 && c1 <= self.filled, "column range out of bounds");
        let (w, chi) = (self.data.shape()[0], self.data.shape()[1]);
        self.data.slice(&[(0, w), (0, chi), (c0, c1)])
    }
}

/// Per-site batched environment storage. `bufs[i]` holds, for every sketch
/// column created so far, the environment absorbing sites `0..=i` — shape
/// `(w, chi)` over the MPO / MPS bonds to the right of site `i`, stacked
/// along a trailing batch axis. Site `j` of the sweep consumes `bufs[j-1]`;
/// because the sweep moves right to left and every column is recursed
/// through `0..=j-1` at creation, the prefix a later (more-left) site needs
/// is always present.
struct EnvStore<T: Scalar> {
    bufs: Vec<Option<EnvBuf<T>>>,
}

impl<T: Scalar> EnvStore<T> {
    /// One slot per site for uniform indexing; the last slot stays `None`
    /// forever, because the recursion stops at site `n - 2` (site `n - 1`,
    /// the sweep's starting point, consumes `bufs[n - 2]`).
    fn new(n: usize) -> Self {
        Self {
            bufs: (0..n).map(|_| None).collect(),
        }
    }

    /// Number of sketch columns created so far. Every column's recursion
    /// starts at site 0, so the first buffer's fill count is the global
    /// column count.
    fn ncols(&self) -> usize {
        self.bufs[0].as_ref().map_or(0, |b| b.filled)
    }

    /// Drop a site's buffer once the sweep has moved past its last reader.
    fn release(&mut self, site: usize) {
        self.bufs[site] = None;
    }

    fn append(&mut self, site: usize, block: &DenseTensorData<T>) {
        match &mut self.bufs[site] {
            Some(buf) => buf.append(block),
            None => {
                self.bufs[site] = Some(EnvBuf {
                    data: block.clone(),
                    filled: block.shape()[2],
                });
            }
        }
    }

    fn range(&self, site: usize, c0: usize, c1: usize) -> DenseTensorData<T> {
        self.bufs[site]
            .as_ref()
            .expect("environments are recursed through a site before it is read")
            .range(c0, c1)
    }
}

/// Create `s` fresh sketch columns and run their environment recursion
/// through sites `0..=last` in one batched contraction per site, appending
/// each site's `(w, chi, s)` block to the store. Gaussians are drawn on the
/// fly and fully absorbed, so only the environments persist.
///
/// Of the pairwise contractions only the environment-carrying step has the
/// batch index on both operands; the Gaussian absorption and the MPS-site
/// contraction fold it into a free GEMM dimension.
fn extend_new_columns<T, B>(
    backend: &B,
    op: &Mpo<DenseStorage<T>, DenseLayout>,
    psi: &Mps<DenseStorage<T>, DenseLayout>,
    envs: &mut EnvStore<T>,
    last: usize,
    s: usize,
    rng: &mut StdRng,
) where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    // Left boundary: dummy (w = 1, chi = 1) environments for every column.
    let mut cur = DenseTensor::<T>::ones(vec![1, 1, s]);
    for i in 0..=last {
        let w = op.site(i);
        let a = psi.site(i);
        let omega = gaussian_block::<T, _>(s, w.shape()[2], rng);
        // (c, b) x (w, k, b, w') x (w, chi, c) x (chi, k, chi') -> (w', chi', c).
        let next = einsum_with_backend(backend, &[&omega, w, &cur, a], "cb,wkbv,wxc,xky->vyc")
            .expect("sketch recursion: MPO/MPS site legs must be compatible");
        envs.append(i, next.data());
        cur = next;
    }
}

/// Batched sketch panels for one column block at site `j`: contract the
/// columns' environments with the site pair and the cap in one pass,
/// yielding a `(d_bra * cap_dim, s)` matrix whose fused row leg matches the
/// `split_leg` applied to the Q factor.
fn sketch_panel_block<T: Scalar, B: OpsFor<DenseStorage<T>>>(
    backend: &B,
    env_block: &DenseTensor<T>,
    w: &DenseTensor<T>,
    a: &DenseTensor<T>,
    cap: &DenseTensor<T>,
) -> DenseTensor<T> {
    // (w, chi, c) x (w, k, b, w') x (chi, k, chi') x (z, w', chi') -> (b, z, c);
    // the batch index rides through as a free GEMM dimension at every step.
    let t = einsum_with_backend(backend, &[env_block, w, a, cap], "wxc,wkbv,xky,zvy->bzc")
        .expect("sketch contraction: MPO/MPS site legs must be compatible");
    t.fuse_legs(0..2)
}

/// Leave-one-out error estimate from the squared row norms of `R^-1`
/// maintained by the incremental QR: `sqrt((1/p) * sum_i ||g_i||^-2)` over
/// the columns `g_i` of `G = R^-dagger` (arXiv:2504.06475) — row norms of
/// `R^-1` are the same quantities. Rank deficiency is reported by the QR
/// append itself ([`QrAppendOutcome::RankDeficient`]) before this runs, so
/// the row norms fed here always come from an invertible factor.
fn leave_one_out_estimate<T: Scalar>(row_sq_norms: &[T::Real]) -> T::Real {
    let p = row_sq_norms.len();
    let mut inv_sq_sum = T::Real::zero();
    for r in row_sq_norms {
        inv_sq_sum = inv_sq_sum + r.recip();
    }
    let p_real = <T::Real as NumCast>::from(p).expect("bond dimensions fit in the real type");
    (inv_sq_sum / p_real).sqrt()
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
    // Adaptive mode reads the estimator off the incremental QR; fixed mode
    // does a single append and never consults it, so it skips the tracking.
    let adaptive = src.output_dim.is_none();

    let mut envs = EnvStore::<T>::new(n);
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
        let mut target_p = match src.output_dim {
            Some(k) => k.min(current_maxdim),
            None => src.sketch_dim.min(current_maxdim).max(current_mindim),
        };

        let mut inc = IncrementalQr::<T>::new(rows, adaptive);
        let mut norm_sq = T::Real::zero();
        loop {
            let columns_created = envs.ncols();
            if target_p > columns_created {
                extend_new_columns(
                    backend,
                    op,
                    psi,
                    &mut envs,
                    j - 1,
                    target_p - columns_created,
                    &mut rng,
                );
            }
            let c0 = inc.ncols();
            let env_block = DenseTensor::from_data(envs.range(j - 1, c0, target_p));
            let panel = sketch_panel_block(backend, &env_block, w, a, &cap);
            // Fixed mode breaks before either consumer of the accumulated
            // norm (the zero-norm break and the estimator), so the panel
            // pass is adaptive-only work.
            if adaptive {
                let panel_norm = panel.norm();
                norm_sq = norm_sq + panel_norm * panel_norm;
            }
            // The `current_maxdim` clamp above keeps the block within the
            // factorization's bounds, so a failure here is an unrecoverable
            // backend error, not a rejected input.
            let outcome = inc.append(backend, &panel).expect(
                "sketch QR append: the clamps keep the block within the row budget, \
                     so only an unrecoverable backend kernel failure lands here",
            );
            let p = inc.ncols();

            if !adaptive || p == current_maxdim {
                break;
            }
            if norm_sq.is_zero() {
                // Zero product at this cut: any orthonormal basis is exact.
                break;
            }
            if outcome == QrAppendOutcome::RankDeficient {
                // Rank-deficient sketch = the exact rank is already covered.
                break;
            }
            let row_sq = inc
                .r_inverse_row_sq_norms()
                .expect("adaptive mode tracks the inverse and this append was full-rank");
            let err = leave_one_out_estimate::<T>(row_sq);
            let p_real =
                <T::Real as NumCast>::from(p).expect("bond dimensions fit in the real type");
            let norm_est = norm_sq.sqrt() / p_real.sqrt();
            let cutoff = cutoff_real.expect("validated: adaptive mode has a cutoff");
            // p started at or above the clamped min_dim and only grows, so
            // the cutoff test alone decides convergence.
            if err <= cutoff * norm_est {
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
        // sketch.
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
        // This iteration was `bufs[j - 1]`'s last reader (site j - 1 reads
        // `bufs[j - 2]`), so its environments can be freed now instead of
        // holding every site's buffer until the sweep ends.
        envs.release(j - 1);
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
