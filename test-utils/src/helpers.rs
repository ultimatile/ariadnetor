//! Test helpers shared across BlockSparse Heff integration test
//! sub-modules: densify a `BlockSparseTensor<T, U1Sector>` to a CM
//! `DenseTensor<T>`, build per-template-block flat offset tables, and
//! convert back and forth between flat-template-aware vectors and
//! Dense rank-4 tensors in the global shape.

use arnet_tensor::MemoryOrder;
use arnet_tensor::{
    BlockSparseTensor, ComputeBackendTensorExt, DenseTensor, Host, Sector, U1Sector,
};
use num_complex::Complex;

/// Build a host-resident `DenseTensor` from a flat buffer already laid
/// out in the host substrate's preferred order, tagging that order.
///
/// Shared fixture constructor for tests that previously reached for the
/// removed `DenseTensor::from_raw_parts`; the `_host` name discloses the
/// order the buffer is read under. Routes through the Mid-layer
/// `make_tensor` surface rather than a raw constructor.
pub fn dense_host<T: Clone>(data: Vec<T>, shape: Vec<usize>) -> DenseTensor<T> {
    DenseTensor::from_data(Host::shared().make_tensor(data, shape))
}

pub fn densify_bsp_f64(bsp: &BlockSparseTensor<f64, U1Sector>) -> DenseTensor<f64> {
    densify_bsp_generic(bsp, 0.0)
}

pub fn densify_bsp_c64(
    bsp: &BlockSparseTensor<Complex<f64>, U1Sector>,
) -> DenseTensor<Complex<f64>> {
    densify_bsp_generic(bsp, Complex::new(0.0, 0.0))
}

fn densify_bsp_generic<T: arnet_core::Scalar>(
    bsp: &BlockSparseTensor<T, U1Sector>,
    zero: T,
) -> DenseTensor<T> {
    let global_dims: Vec<usize> = bsp.shape().to_vec();
    let total: usize = global_dims.iter().product();
    let mut out = vec![zero; total];
    let rank = global_dims.len();
    let prefix_offsets: Vec<Vec<usize>> = bsp
        .indices()
        .iter()
        .map(|idx| {
            let mut acc = 0usize;
            (0..idx.num_blocks())
                .map(|b| {
                    let cur = acc;
                    acc += idx.block_dim(b);
                    cur
                })
                .collect()
        })
        .collect();
    for meta in bsp.block_metas() {
        let coord = &meta.coord;
        let block_shape = bsp.block_shape(coord).expect("allowed block");
        let block_data = bsp.block_data(coord).expect("allowed block");
        let offsets: Vec<usize> = (0..rank)
            .map(|axis| prefix_offsets[axis][coord.0[axis]])
            .collect();
        let block_total: usize = block_shape.iter().product();
        let mut local = vec![0_usize; rank];
        for _ in 0..block_total {
            // Per-block CM flat index (matches the BlockSparse
            // backend's preferred order on this platform).
            let mut cm_flat = 0_usize;
            let mut stride = 1_usize;
            for axis in 0..rank {
                cm_flat += local[axis] * stride;
                stride *= block_shape[axis];
            }
            // CM global index for the scatter buffer, matching the
            // backend's preferred order so no reorder is needed.
            let mut g = 0_usize;
            let mut g_stride = 1_usize;
            for axis in 0..rank {
                g += (offsets[axis] + local[axis]) * g_stride;
                g_stride *= global_dims[axis];
            }
            out[g] = block_data[cm_flat];
            for axis in (0..rank).rev() {
                local[axis] += 1;
                if local[axis] < block_shape[axis] {
                    break;
                }
                local[axis] = 0;
            }
        }
    }
    dense_host(out, global_dims)
}

pub fn template_block_offsets(template: &BlockSparseTensor<f64, U1Sector>) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(template.num_blocks() + 1);
    let mut running = 0_usize;
    for meta in template.block_metas() {
        offsets.push(running);
        running += template
            .block_shape(&meta.coord)
            .expect("template block")
            .iter()
            .product::<usize>();
    }
    offsets.push(running);
    offsets
}

pub fn template_from_mps_pair(
    mps_i: &BlockSparseTensor<f64, U1Sector>,
    mps_ip1: &BlockSparseTensor<f64, U1Sector>,
) -> BlockSparseTensor<f64, U1Sector> {
    let psi_indices = vec![
        mps_i.indices()[0].clone(),
        mps_i.indices()[1].clone(),
        mps_ip1.indices()[1].clone(),
        mps_ip1.indices()[2].clone(),
    ];
    let psi_flux = mps_i.flux().fuse(mps_ip1.flux());
    BlockSparseTensor::<f64, U1Sector>::zeros(psi_indices, psi_flux)
}

/// Densified rank-4 → flat-template-aware vec. Used to build the
/// "expected" output of the BlockSparse matvec given a Dense matvec
/// applied to densified inputs.
pub fn dense_to_template_flat(
    dense: &DenseTensor<f64>,
    template: &BlockSparseTensor<f64, U1Sector>,
) -> Vec<f64> {
    let global_dims: Vec<usize> = dense.shape().to_vec();
    let rank = global_dims.len();
    let prefix_offsets: Vec<Vec<usize>> = template
        .indices()
        .iter()
        .map(|idx| {
            let mut acc = 0usize;
            (0..idx.num_blocks())
                .map(|b| {
                    let cur = acc;
                    acc += idx.block_dim(b);
                    cur
                })
                .collect()
        })
        .collect();
    let dense_rm = dense.reordered(MemoryOrder::RowMajor);
    let dense_data = dense_rm.data_slice();
    let block_offsets = template_block_offsets(template);
    let total_dim = *block_offsets.last().unwrap_or(&0);
    let mut flat = vec![0.0_f64; total_dim];
    for (k, meta) in template.block_metas().iter().enumerate() {
        let coord = &meta.coord;
        let block_shape = template.block_shape(coord).expect("template block");
        let offsets: Vec<usize> = (0..rank)
            .map(|axis| prefix_offsets[axis][coord.0[axis]])
            .collect();
        let lo = block_offsets[k];
        let block_total: usize = block_shape.iter().product();
        let mut local = vec![0_usize; rank];
        for _ in 0..block_total {
            let mut cm_flat = 0_usize;
            let mut stride = 1_usize;
            for axis in 0..rank {
                cm_flat += local[axis] * stride;
                stride *= block_shape[axis];
            }
            let mut g = 0_usize;
            for axis in 0..rank {
                g = g * global_dims[axis] + (offsets[axis] + local[axis]);
            }
            flat[lo + cm_flat] = dense_data[g];
            for axis in (0..rank).rev() {
                local[axis] += 1;
                if local[axis] < block_shape[axis] {
                    break;
                }
                local[axis] = 0;
            }
        }
    }
    flat
}

/// Build the dense `psi[chi_l, d_i, d_{i+1}, chi_r]` by scattering
/// a template-flat slice into the right global positions, returning
/// a Dense in CM (NativeBackend preferred order).
pub fn build_dense_psi_from_flat(
    flat: &[f64],
    template: &BlockSparseTensor<f64, U1Sector>,
) -> DenseTensor<f64> {
    let global_dims: Vec<usize> = template.shape().to_vec();
    let total: usize = global_dims.iter().product();
    let rank = global_dims.len();
    let mut cm_data = vec![0.0_f64; total];
    let prefix_offsets: Vec<Vec<usize>> = template
        .indices()
        .iter()
        .map(|idx| {
            let mut acc = 0usize;
            (0..idx.num_blocks())
                .map(|b| {
                    let cur = acc;
                    acc += idx.block_dim(b);
                    cur
                })
                .collect()
        })
        .collect();
    let block_offsets = template_block_offsets(template);
    for (k, meta) in template.block_metas().iter().enumerate() {
        let coord = &meta.coord;
        let block_shape = template.block_shape(coord).expect("template block");
        let offsets: Vec<usize> = (0..rank)
            .map(|axis| prefix_offsets[axis][coord.0[axis]])
            .collect();
        let lo = block_offsets[k];
        let block_total: usize = block_shape.iter().product();
        let mut local = vec![0_usize; rank];
        for _ in 0..block_total {
            let mut cm_flat = 0_usize;
            let mut stride = 1_usize;
            for axis in 0..rank {
                cm_flat += local[axis] * stride;
                stride *= block_shape[axis];
            }
            let mut g = 0_usize;
            let mut g_stride = 1_usize;
            for axis in 0..rank {
                g += (offsets[axis] + local[axis]) * g_stride;
                g_stride *= global_dims[axis];
            }
            cm_data[g] = flat[lo + cm_flat];
            for axis in (0..rank).rev() {
                local[axis] += 1;
                if local[axis] < block_shape[axis] {
                    break;
                }
                local[axis] = 0;
            }
        }
    }
    dense_host(cm_data, global_dims)
}
