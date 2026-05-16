//! Construction APIs for [`BlockSparse`] and its mutable-block access.

use std::collections::HashMap;
use std::sync::Arc;

use aligned_vec::{AVec, ConstAlign};
use arnet_core::backend::MemoryOrder;

use super::{BlockCoord, BlockMeta, BlockSparse, QNIndex};
use crate::sector::Sector;

impl<T: Clone, S: Sector> BlockSparse<T, S> {
    /// Construct a `BlockSparse` from pre-validated components.
    ///
    /// Performs consistency checks (coord validity, flux conservation,
    /// offset/size bounds, contiguous packing) but does NOT enumerate
    /// allowed blocks — the caller is responsible for providing correct blocks.
    ///
    /// # Panics
    ///
    /// Panics if any invariant is violated.
    #[cfg(test)]
    pub(crate) fn from_raw_parts(
        data: Vec<T>,
        blocks: Vec<BlockMeta>,
        indices: Vec<QNIndex<S>>,
        flux: S,
    ) -> Self {
        let rank = indices.len();
        let shape: Vec<usize> = indices.iter().map(|idx| idx.total_dim()).collect();

        // Validate blocks
        let mut block_index = HashMap::with_capacity(blocks.len());
        for (i, meta) in blocks.iter().enumerate() {
            assert_eq!(
                meta.coord.0.len(),
                rank,
                "BlockMeta coord rank mismatch: expected {rank}, got {}",
                meta.coord.0.len()
            );

            // Validate coord bounds
            for (axis, &block_idx) in meta.coord.0.iter().enumerate() {
                assert!(
                    block_idx < indices[axis].num_blocks(),
                    "Block index {block_idx} out of range for axis {axis} (max {})",
                    indices[axis].num_blocks()
                );
            }

            // Validate size matches block shape
            let expected_size: usize = meta
                .coord
                .0
                .iter()
                .enumerate()
                .map(|(axis, &bi)| indices[axis].block_dim(bi))
                .product();
            assert_eq!(
                meta.size, expected_size,
                "BlockMeta size mismatch at coord {:?}: expected {expected_size}, got {}",
                meta.coord, meta.size
            );

            // Validate offset + size within data bounds
            assert!(
                meta.offset + meta.size <= data.len(),
                "BlockMeta at coord {:?} exceeds data buffer: offset {} + size {} > len {}",
                meta.coord,
                meta.offset,
                meta.size,
                data.len()
            );

            // Validate flux conservation
            let mut fused = S::identity();
            for (axis, &bi) in meta.coord.0.iter().enumerate() {
                let sector = indices[axis].sector(bi);
                let directed = indices[axis].direction().apply(sector);
                fused = fused.fuse(&directed);
            }
            assert_eq!(
                fused, flux,
                "Block {:?} violates flux conservation: fused {:?} != flux {:?}",
                meta.coord, fused, flux
            );

            let prev = block_index.insert(meta.coord.clone(), i);
            assert!(
                prev.is_none(),
                "Duplicate block coordinate {:?}",
                meta.coord
            );
        }

        // Verify blocks are sorted by coord
        for i in 1..blocks.len() {
            assert!(
                blocks[i - 1].coord < blocks[i].coord,
                "Blocks not sorted: {:?} >= {:?}",
                blocks[i - 1].coord,
                blocks[i].coord
            );
        }

        // Verify blocks tile the data buffer contiguously without gaps or overlaps
        let mut expected_offset = 0;
        // Sort by offset to verify packing (blocks are sorted by coord, not offset)
        let mut offset_order: Vec<usize> = (0..blocks.len()).collect();
        offset_order.sort_by_key(|&i| blocks[i].offset);
        for &i in &offset_order {
            assert_eq!(
                blocks[i].offset, expected_offset,
                "Block {:?} has offset {} but expected {} (gap or overlap)",
                blocks[i].coord, blocks[i].offset, expected_offset
            );
            expected_offset += blocks[i].size;
        }
        assert_eq!(
            expected_offset,
            data.len(),
            "Data buffer has {} elements but blocks cover only {}",
            data.len(),
            expected_offset
        );

        let aligned_data = AVec::from_slice(64, &data);

        Self {
            data: Arc::new(aligned_data),
            blocks,
            block_index,
            indices,
            flux,
            shape,
            order: MemoryOrder::ColumnMajor,
        }
    }
}

impl<S: Sector> BlockSparse<(), S> {
    /// Enumerate flux-allowed blocks and build structural metadata.
    ///
    /// Returns `(blocks, block_index, shape, total_size)` — everything needed
    /// to construct a `BlockSparse` except the data buffer itself.
    pub(super) fn build_structure(
        indices: &[QNIndex<S>],
        flux: &S,
    ) -> (
        Vec<BlockMeta>,
        HashMap<BlockCoord, usize>,
        Vec<usize>,
        usize,
    ) {
        let shape: Vec<usize> = indices.iter().map(|idx| idx.total_dim()).collect();
        let rank = indices.len();
        let num_blocks_per_leg: Vec<usize> = indices.iter().map(|idx| idx.num_blocks()).collect();

        let mut blocks = Vec::new();
        let mut total_size = 0usize;

        // Enumerate all block coordinate combinations in lexicographic order.
        // Skip if any leg has no sectors (tensor is empty).
        if rank == 0 || num_blocks_per_leg.iter().all(|&n| n > 0) {
            let mut current = vec![0usize; rank];
            loop {
                // Check flux conservation for this coordinate
                let mut fused = S::identity();
                for (axis, &bi) in current.iter().enumerate() {
                    let sector = indices[axis].sector(bi);
                    let directed = indices[axis].direction().apply(sector);
                    fused = fused.fuse(&directed);
                }

                if fused == *flux {
                    let size: usize = current
                        .iter()
                        .enumerate()
                        .map(|(axis, &bi)| indices[axis].block_dim(bi))
                        .product();
                    blocks.push(BlockMeta {
                        coord: BlockCoord(current.clone()),
                        offset: total_size,
                        size,
                    });
                    total_size += size;
                }

                // Increment multi-index (lexicographic order)
                let mut carry = true;
                for axis in (0..rank).rev() {
                    current[axis] += 1;
                    if current[axis] < num_blocks_per_leg[axis] {
                        carry = false;
                        break;
                    }
                    current[axis] = 0;
                }
                if carry {
                    break;
                }
            }
        }

        let mut block_index = HashMap::with_capacity(blocks.len());
        for (i, meta) in blocks.iter().enumerate() {
            block_index.insert(meta.coord.clone(), i);
        }

        (blocks, block_index, shape, total_size)
    }
}

impl<T, S: Sector> BlockSparse<T, S> {
    /// Construct a zero-filled `BlockSparse` with all flux-allowed blocks.
    ///
    /// Enumerates every block coordinate satisfying the flux conservation law
    /// and allocates a contiguous zero-filled data buffer.
    pub fn zeros(indices: Vec<QNIndex<S>>, flux: S) -> Self
    where
        T: Clone + num_traits::Zero,
    {
        let (blocks, block_index, shape, total_size) =
            BlockSparse::<(), S>::build_structure(&indices, &flux);

        let mut data = AVec::<T, ConstAlign<64>>::with_capacity(64, total_size);
        data.resize(total_size, T::zero());

        Self {
            data: Arc::new(data),
            blocks,
            block_index,
            indices,
            flux,
            shape,
            order: MemoryOrder::ColumnMajor,
        }
    }

    /// Construct a `BlockSparse` with all flux-allowed blocks filled with
    /// random values from the standard distribution.
    ///
    /// The tensor structure (shape, blocks, flux) is identical to [`Self::zeros`];
    /// only the data differs.
    pub fn random<R: rand::Rng>(indices: Vec<QNIndex<S>>, flux: S, rng: &mut R) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let (blocks, block_index, shape, total_size) =
            BlockSparse::<(), S>::build_structure(&indices, &flux);

        let mut data = AVec::<T, ConstAlign<64>>::with_capacity(64, total_size);
        for _ in 0..total_size {
            data.push(rng.random());
        }

        Self {
            data: Arc::new(data),
            blocks,
            block_index,
            indices,
            flux,
            shape,
            order: MemoryOrder::ColumnMajor,
        }
    }

    /// Construct a `BlockSparse` by initializing each flux-allowed block via
    /// a caller-supplied closure, without a prior zero-write of the data buffer.
    ///
    /// The closure receives the block's coordinate and per-leg shape and must
    /// return a `Vec<T>` of exactly `block_shape.iter().product()` elements in
    /// the layout declared by `source_order`.
    /// The constructor invokes the closure exactly once per flux-allowed block,
    /// in lexicographic coordinate order.
    ///
    /// `source_order` is stored on the resulting tensor (see [`Self::order`]).
    /// The closure's returned bytes are appended to the buffer verbatim — no
    /// internal reorder is performed.
    ///
    /// # Current consumer-side limitation
    ///
    /// Existing block-sparse linalg paths (permute, contract, decomp) read raw
    /// block data under `backend.preferred_order()` rather than consulting
    /// `tensor.order()`, so a tensor built with `source_order = RowMajor` is
    /// not yet honored by those consumers. Pass `MemoryOrder::ColumnMajor` to
    /// match the current convention; the field is exposed in preparation for
    /// the consumer-side fix cascade (analogous to the `Dense` evolution).
    ///
    /// # Panics
    ///
    /// Panics if the closure returns a `Vec<T>` whose length differs from the
    /// block's product-of-dimensions.
    pub fn from_block_fn<F>(
        indices: Vec<QNIndex<S>>,
        flux: S,
        source_order: MemoryOrder,
        mut fill: F,
    ) -> Self
    where
        F: FnMut(&BlockCoord, &[usize]) -> Vec<T>,
    {
        let (blocks, block_index, shape, total_size) =
            BlockSparse::<(), S>::build_structure(&indices, &flux);

        let mut data = AVec::<T, ConstAlign<64>>::with_capacity(64, total_size);

        for meta in &blocks {
            let block_shape: Vec<usize> = meta
                .coord
                .0
                .iter()
                .enumerate()
                .map(|(axis, &bi)| indices[axis].block_dim(bi))
                .collect();
            let block_data = fill(&meta.coord, &block_shape);
            assert_eq!(
                block_data.len(),
                meta.size,
                "BlockSparse::from_block_fn: closure returned {} elements for block {:?}, expected {}",
                block_data.len(),
                meta.coord,
                meta.size
            );
            for v in block_data {
                data.push(v);
            }
        }

        debug_assert_eq!(
            data.len(),
            total_size,
            "BlockSparse::from_block_fn: data length {} != total_size {} (build_structure packing invariant violated)",
            data.len(),
            total_size
        );

        Self {
            data: Arc::new(data),
            blocks,
            block_index,
            indices,
            flux,
            shape,
            order: source_order,
        }
    }

    /// Mutable data slice for a block identified by coordinate (CoW).
    ///
    /// Triggers a copy of the entire data buffer if other clones exist.
    /// Returns `None` if the block is not stored.
    pub fn block_data_mut(&mut self, coord: &BlockCoord) -> Option<&mut [T]>
    where
        T: Clone,
    {
        let &idx = self.block_index.get(coord)?;
        let meta = &self.blocks[idx];
        let offset = meta.offset;
        let size = meta.size;
        let data = Arc::make_mut(&mut self.data);
        Some(&mut data[offset..offset + size])
    }
}
