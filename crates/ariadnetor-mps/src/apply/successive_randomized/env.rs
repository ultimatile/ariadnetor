//! Sketch-environment machinery for successive randomized compression:
//! per-site Gaussian blocks, batched per-term environment buffers, and
//! the per-site sketch-panel contraction. The parent module doc carries
//! the algorithm narrative; this module owns the buffer plumbing the
//! sweep consumes.

use ariadnetor_core::Scalar;
use ariadnetor_linalg::einsum_with_backend;
use ariadnetor_tensor::{DenseStorage, DenseTensor, DenseTensorData, OpsFor};
use num_traits::{NumCast, Zero};
use rand::RngExt;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

use super::DenseTerm;
use crate::chain::TensorChain;

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

/// Per-site batched environment storage for one term. `bufs[i]` holds, for
/// every sketch column created so far, the environment absorbing sites
/// `0..=i` — shape `(w, chi)` over the term's MPO / MPS bonds to the right
/// of site `i`, stacked along a trailing batch axis. Site `j` of the sweep
/// consumes `bufs[j-1]`; because the sweep moves right to left and every
/// column is recursed through `0..=j-1` at creation, the prefix a later
/// (more-left) site needs is always present.
pub(super) struct EnvStore<T: Scalar> {
    bufs: Vec<Option<EnvBuf<T>>>,
}

impl<T: Scalar> EnvStore<T> {
    /// One slot per site for uniform indexing; the last slot stays `None`
    /// forever, because the recursion stops at site `n - 2` (site `n - 1`,
    /// the sweep's starting point, consumes `bufs[n - 2]`).
    pub(super) fn new(n: usize) -> Self {
        Self {
            bufs: (0..n).map(|_| None).collect(),
        }
    }

    /// Number of sketch columns created so far. Every column's recursion
    /// starts at site 0, so the first buffer's fill count is the global
    /// column count.
    pub(super) fn ncols(&self) -> usize {
        self.bufs[0].as_ref().map_or(0, |b| b.filled)
    }

    /// Drop a site's buffer once the sweep has moved past its last reader.
    pub(super) fn release(&mut self, site: usize) {
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

    pub(super) fn range(&self, site: usize, c0: usize, c1: usize) -> DenseTensorData<T> {
        self.bufs[site]
            .as_ref()
            .expect("environments are recursed through a site before it is read")
            .range(c0, c1)
    }
}

/// Create `s` fresh sketch columns and run their environment recursion
/// through sites `0..=last` for every term, appending each site's
/// `(w_t, chi_t, s)` block to that term's store. Each site's Gaussian
/// block is drawn once and reused across the terms' recursions: the test
/// matrix lives on the shared output physical legs, so reusing the block
/// is what makes the summed panel a sketch of the summed operand (see the
/// module doc). Gaussians are drawn on the fly and fully absorbed, so
/// only the environments persist. With one term the draw order is
/// bit-identical to the pre-generalization kernel (site-major, one block
/// per site).
///
/// Of the pairwise contractions only the environment-carrying step has the
/// batch index on both operands; the Gaussian absorption and the MPS-site
/// contraction fold it into a free GEMM dimension.
pub(super) fn extend_new_columns<T, B>(
    backend: &B,
    terms: &[DenseTerm<'_, T>],
    envs: &mut [EnvStore<T>],
    last: usize,
    s: usize,
    rng: &mut StdRng,
) where
    T: Scalar,
    B: OpsFor<DenseStorage<T>>,
{
    // Left boundary: dummy (w = 1, chi = 1) environments for every column,
    // one running environment per term.
    let mut curs: Vec<DenseTensor<T>> = (0..terms.len())
        .map(|_| DenseTensor::<T>::ones(vec![1, 1, s]))
        .collect();
    for i in 0..=last {
        // Output (bra) dimensions agree across terms (validated at entry),
        // so one Gaussian block serves every term's recursion at this site.
        let omega = gaussian_block::<T, _>(s, terms[0].0.site(i).shape()[2], rng);
        for (t, (op, psi)) in terms.iter().enumerate() {
            let w = op.site(i);
            let a = psi.site(i);
            // (c, b) x (w, k, b, w') x (w, chi, c) x (chi, k, chi') -> (w', chi', c).
            let next =
                einsum_with_backend(backend, &[&omega, w, &curs[t], a], "cb,wkbv,wxc,xky->vyc")
                    .expect("sketch recursion: MPO/MPS site legs must be compatible");
            envs[t].append(i, next.data());
            curs[t] = next;
        }
    }
}

/// Batched sketch panels for one column block of one term at site `j`:
/// contract the columns' environments with the term's site pair and cap in
/// one pass, yielding a `(d_bra * cap_dim, s)` matrix whose fused row leg
/// matches the `split_leg` applied to the Q factor. Every term's panel has
/// the same row count — `d_bra` is validated equal across terms and
/// `cap_dim` is the shared Q column count — so the panels can be summed.
pub(super) fn sketch_panel_block<T: Scalar, B: OpsFor<DenseStorage<T>>>(
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
