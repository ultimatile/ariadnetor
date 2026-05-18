mod construction;
mod tensor_data;

use super::*;
use crate::sector::{U1Sector, Z2Sector};

// ---------------------------------------------------------------------------
// Direction
// ---------------------------------------------------------------------------

#[test]
fn direction_apply_out() {
    let s = U1Sector(3);
    assert_eq!(Direction::Out.apply(&s), U1Sector(3));
}

#[test]
fn direction_apply_in() {
    let s = U1Sector(3);
    assert_eq!(Direction::In.apply(&s), U1Sector(-3));
}

// ---------------------------------------------------------------------------
// QNIndex
// ---------------------------------------------------------------------------

#[test]
fn qnindex_sorts_blocks() {
    // Provide unsorted input; constructor should sort
    let idx = QNIndex::new(
        vec![(U1Sector(2), 3), (U1Sector(-1), 2), (U1Sector(0), 1)],
        Direction::Out,
    );
    let sectors: Vec<_> = idx.blocks().iter().map(|(s, _)| s.0).collect();
    assert_eq!(sectors, vec![-1, 0, 2]);
}

#[test]
fn qnindex_accessors() {
    let idx = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    assert_eq!(idx.num_blocks(), 2);
    assert_eq!(idx.total_dim(), 5);
    assert_eq!(idx.block_dim(0), 2);
    assert_eq!(idx.block_dim(1), 3);
    assert_eq!(*idx.sector(0), U1Sector(0));
    assert_eq!(*idx.sector(1), U1Sector(1));
    assert_eq!(idx.direction(), Direction::In);
}

#[test]
#[should_panic(expected = "block dimension must be > 0")]
fn qnindex_rejects_zero_dim() {
    QNIndex::new(vec![(U1Sector(0), 0)], Direction::Out);
}

#[test]
#[should_panic(expected = "duplicate sector")]
fn qnindex_rejects_duplicates() {
    QNIndex::new(vec![(U1Sector(1), 2), (U1Sector(1), 3)], Direction::Out);
}

#[test]
fn qnindex_single_sector() {
    // Trivial leg (e.g., MPS boundary)
    let idx = QNIndex::new(vec![(U1Sector(0), 1)], Direction::Out);
    assert_eq!(idx.num_blocks(), 1);
    assert_eq!(idx.total_dim(), 1);
}

#[test]
fn qnindex_z2() {
    let idx = QNIndex::new(
        vec![(Z2Sector::new(1), 4), (Z2Sector::new(0), 3)],
        Direction::Out,
    );
    // Should be sorted: Z2(0) < Z2(1)
    assert_eq!(*idx.sector(0), Z2Sector::new(0));
    assert_eq!(idx.block_dim(0), 3);
    assert_eq!(*idx.sector(1), Z2Sector::new(1));
    assert_eq!(idx.block_dim(1), 4);
}

// ---------------------------------------------------------------------------
// BlockCoord
// ---------------------------------------------------------------------------

#[test]
fn block_coord_ord() {
    let a = BlockCoord(vec![0, 1]);
    let b = BlockCoord(vec![1, 0]);
    assert!(a < b);
}

#[test]
fn block_coord_eq_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(BlockCoord(vec![0, 1]));
    assert!(set.contains(&BlockCoord(vec![0, 1])));
    assert!(!set.contains(&BlockCoord(vec![1, 0])));
}

// ---------------------------------------------------------------------------
// BlockSparseTensorData — joined accessors and structural invariants
// ---------------------------------------------------------------------------

use arnet_core::backend::MemoryOrder;

/// Helper: rank-2 BlockSparseTensorData with U(1) symmetry, flux 0,
/// allowed blocks (0,0) of size 2×2 with values 1..=4 and (1,1) of
/// size 3×3 with values 5..=13.
pub(super) fn sample_u1_rank2_data() -> BlockSparseTensorData<f64, U1Sector> {
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::In);
    let mut td = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(0),
        MemoryOrder::RowMajor,
    );
    {
        let d = td.block_data_mut(&BlockCoord(vec![0, 0])).unwrap();
        d.copy_from_slice(&(1..=4).map(|x| x as f64).collect::<Vec<_>>());
    }
    {
        let d = td.block_data_mut(&BlockCoord(vec![1, 1])).unwrap();
        d.copy_from_slice(&(5..=13).map(|x| x as f64).collect::<Vec<_>>());
    }
    td
}

#[test]
fn block_sparse_basic_accessors() {
    let td = sample_u1_rank2_data();
    assert_eq!(td.rank(), 2);
    assert_eq!(td.shape(), vec![5, 5]);
    assert_eq!(td.num_blocks(), 2);
    assert_eq!(td.storage().stored_len(), 13);
    assert_eq!(*td.flux(), U1Sector(0));
}

#[test]
fn block_sparse_block_data() {
    let td = sample_u1_rank2_data();

    let d00 = td.block_data(&BlockCoord(vec![0, 0])).unwrap();
    assert_eq!(d00, &[1.0, 2.0, 3.0, 4.0]);

    let d11 = td.block_data(&BlockCoord(vec![1, 1])).unwrap();
    assert_eq!(d11, &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);

    // Non-existent block (forbidden by symmetry)
    assert!(td.block_data(&BlockCoord(vec![0, 1])).is_none());
    assert!(td.block_data(&BlockCoord(vec![1, 0])).is_none());
}

#[test]
fn block_sparse_block_shape() {
    let td = sample_u1_rank2_data();
    assert_eq!(td.block_shape(&BlockCoord(vec![0, 0])), Some(vec![2, 2]));
    assert_eq!(td.block_shape(&BlockCoord(vec![1, 1])), Some(vec![3, 3]));
    assert_eq!(td.block_shape(&BlockCoord(vec![0, 1])), Some(vec![2, 3]));

    // Invalid coord (out of range)
    assert_eq!(td.block_shape(&BlockCoord(vec![2, 0])), None);
    // Wrong rank
    assert_eq!(td.block_shape(&BlockCoord(vec![0])), None);
}

#[test]
fn block_sparse_is_allowed_block() {
    let td = sample_u1_rank2_data();
    assert!(td.is_allowed_block(&BlockCoord(vec![0, 0])));
    assert!(td.is_allowed_block(&BlockCoord(vec![1, 1])));
    assert!(!td.is_allowed_block(&BlockCoord(vec![0, 1])));
    assert!(!td.is_allowed_block(&BlockCoord(vec![1, 0])));
}

#[test]
fn block_sparse_clone_preserves_shape_and_blocks() {
    let td = sample_u1_rank2_data();
    let cloned = td.clone();
    assert_eq!(cloned.shape(), td.shape());
    assert_eq!(cloned.num_blocks(), td.num_blocks());
    // Block data is preserved (Arc-shared on the storage half).
    assert_eq!(
        cloned.block_data(&BlockCoord(vec![0, 0])).unwrap(),
        td.block_data(&BlockCoord(vec![0, 0])).unwrap(),
    );
}

#[test]
fn block_sparse_nonzero_flux() {
    // Tensor with flux = U1(1)
    // Block (1,0): Out(1) + In(0).dual() = 1 + 0 = 1 == flux ✓
    let row = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 3)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);
    let td = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![row, col],
        U1Sector(1),
        MemoryOrder::RowMajor,
    );
    assert_eq!(td.num_blocks(), 1);
    assert_eq!(*td.flux(), U1Sector(1));
    assert!(td.is_allowed_block(&BlockCoord(vec![1, 0])));
    assert!(!td.is_allowed_block(&BlockCoord(vec![0, 0])));
}

#[test]
fn block_sparse_z2_symmetry() {
    let row = QNIndex::new(
        vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 3)],
        Direction::Out,
    );
    let col = QNIndex::new(
        vec![(Z2Sector::new(0), 2), (Z2Sector::new(1), 3)],
        Direction::In,
    );

    // Z2 is self-dual: allowed blocks (0,0) and (1,1) both fuse to 0
    let td = BlockSparseTensorData::<f64, Z2Sector>::zeros(
        vec![row, col],
        Z2Sector::new(0),
        MemoryOrder::RowMajor,
    );
    assert_eq!(td.num_blocks(), 2);
    assert!(td.is_allowed_block(&BlockCoord(vec![0, 0])));
    assert!(td.is_allowed_block(&BlockCoord(vec![1, 1])));
    assert!(!td.is_allowed_block(&BlockCoord(vec![0, 1])));
}

#[test]
fn block_sparse_empty() {
    let row = QNIndex::new(vec![(U1Sector(0), 2)], Direction::Out);
    let col = QNIndex::new(vec![(U1Sector(0), 2)], Direction::In);

    // Flux U1(1) is unreachable with both legs carrying only charge 0,
    // so no blocks are allowed; logical shape is still 2×2.
    let td: BlockSparseTensorData<f64, U1Sector> =
        BlockSparseTensorData::zeros(vec![row, col], U1Sector(1), MemoryOrder::RowMajor);
    assert_eq!(td.num_blocks(), 0);
    assert_eq!(td.storage().stored_len(), 0);
    assert_eq!(td.shape(), vec![2, 2]);
}

#[test]
fn block_sparse_rank3() {
    // Rank-3: flux = 0
    // (0,0,0): Out(0) + Out(0) + In(0).dual() = 0  size = 2*3*2 = 12
    // (1,0,1): Out(1) + Out(0) + In(1).dual() = 1+0+(-1) = 0  size = 1*3*1 = 3
    let leg0 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::Out);
    let leg1 = QNIndex::new(vec![(U1Sector(0), 3)], Direction::Out);
    let leg2 = QNIndex::new(vec![(U1Sector(0), 2), (U1Sector(1), 1)], Direction::In);

    let td = BlockSparseTensorData::<f64, U1Sector>::zeros(
        vec![leg0, leg1, leg2],
        U1Sector(0),
        MemoryOrder::RowMajor,
    );
    assert_eq!(td.rank(), 3);
    assert_eq!(td.shape(), vec![3, 3, 3]);
    assert_eq!(td.num_blocks(), 2);
    assert_eq!(td.storage().stored_len(), 15);
}

#[test]
fn block_sparse_tuple_symmetry() {
    // U(1) × Z2 direct-product symmetry
    type Sym = (U1Sector, Z2Sector);
    let s00 = (U1Sector(0), Z2Sector::new(0));
    let s11 = (U1Sector(1), Z2Sector::new(1));

    let row = QNIndex::new(vec![(s00, 2), (s11, 3)], Direction::Out);
    let col = QNIndex::new(vec![(s00, 2), (s11, 3)], Direction::In);

    // flux = identity = (0, 0)
    // Block (0,0): Out(0,0) fuse In(0,0).dual() = (0,0) ✓
    // Block (1,1): Out(1,1) fuse In(1,1).dual() = (1,1).fuse((-1,1)) = (0,0) ✓
    let flux: Sym = Sector::identity();
    let td = BlockSparseTensorData::<f64, Sym>::zeros(vec![row, col], flux, MemoryOrder::RowMajor);
    assert_eq!(td.num_blocks(), 2);
}
