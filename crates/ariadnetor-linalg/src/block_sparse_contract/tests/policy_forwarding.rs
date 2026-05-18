//! Verify that the two-layer contract API forwards `ExecPolicy` from the
//! BSp wrapper down to every per-sector GEMM descriptor.
//!
//! `contract_block_sparse` (default) must hardcode `Sequential`;
//! `contract_block_sparse_with_policy` must forward the caller's policy
//! to each per-sector GEMM. The multi-sector fixture produces two block
//! pairs so per-sector forwarding is observable, not just the first hit.

use arnet_core::backend::ExecPolicy;
use arnet_tensor::{BlockCoord, BlockSparseTensorData, Direction, MemoryOrder, QNIndex, U1Sector};

use super::super::{contract_block_sparse, contract_block_sparse_with_policy};
use super::to_order;
use crate::test_util::RecordingBackend;

fn multi_sector_pair() -> (
    BlockSparseTensorData<f64, U1Sector>,
    BlockSparseTensorData<f64, U1Sector>,
) {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::In);
    let mut a = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row.clone(), col.clone()],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    a.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
    a.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[5.0]);
    let mut c = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::ColumnMajor,
    );
    c.block_data_mut(&BlockCoord(vec![0, 0]))
        .unwrap()
        .copy_from_slice(&to_order(&[6.0, 7.0, 8.0, 9.0], &[2, 2]));
    c.block_data_mut(&BlockCoord(vec![1, 1]))
        .unwrap()
        .copy_from_slice(&[10.0]);
    (a, c)
}

#[test]
fn contract_default_forwards_sequential_per_sector_gemm() {
    let rec = RecordingBackend::new();
    let (a, c) = multi_sector_pair();
    let _ = contract_block_sparse(&rec, &a, &c, &[1], &[0]).unwrap();
    let policies = rec.gemm_recorded();
    assert!(
        !policies.is_empty(),
        "expected at least one per-sector GEMM"
    );
    for (i, p) in policies.iter().enumerate() {
        assert_eq!(
            *p,
            ExecPolicy::Sequential,
            "contract default: GEMM #{i} forwarded {p:?}"
        );
    }
}

#[test]
fn contract_with_policy_forwards_parallel_per_sector_gemm() {
    let rec = RecordingBackend::new();
    let (a, c) = multi_sector_pair();
    let _ = contract_block_sparse_with_policy(&rec, &a, &c, &[1], &[0], ExecPolicy::Parallel(0))
        .unwrap();
    let policies = rec.gemm_recorded();
    assert!(
        !policies.is_empty(),
        "expected at least one per-sector GEMM"
    );
    for (i, p) in policies.iter().enumerate() {
        assert_eq!(
            *p,
            ExecPolicy::Parallel(0),
            "contract_with_policy: GEMM #{i} forwarded {p:?}"
        );
    }
}

#[test]
fn contract_with_policy_reaches_every_sector_gemm() {
    let rec = RecordingBackend::new();
    let (a, c) = multi_sector_pair();
    let _ = contract_block_sparse_with_policy(&rec, &a, &c, &[1], &[0], ExecPolicy::Parallel(0))
        .unwrap();
    // Two distinct block pairs — (0,0)×(0,0) and (1,1)×(1,1) — produce two GEMMs.
    assert_eq!(
        rec.gemm_recorded().len(),
        2,
        "expected one GEMM per block-pair sector"
    );
}
