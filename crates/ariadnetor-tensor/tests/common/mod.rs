//! Helpers shared by the dense strip-copy integration tests.

use ariadnetor_tensor::{DenseTensorData, MemoryOrder};

/// `shape`-shaped tensor in `order` whose flat buffer holds `1, 2, 3, …`.
pub(crate) fn ramp(shape: &[usize], order: MemoryOrder) -> DenseTensorData<f64> {
    let n = u32::try_from(shape.iter().product::<usize>()).unwrap();
    DenseTensorData::from_raw_parts((1..=n).map(f64::from).collect(), shape.to_vec(), order)
}

pub(crate) const ORDERS: [MemoryOrder; 2] = [MemoryOrder::RowMajor, MemoryOrder::ColumnMajor];

/// Every multi-index of `shape`, last axis fastest.
pub(crate) fn multi_indices(shape: &[usize]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for &n in shape {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                (0..n).map(move |i| {
                    let mut idx = prefix.clone();
                    idx.push(i);
                    idx
                })
            })
            .collect();
    }
    out
}

/// `idx - offset` per axis when it lands inside `extent`, else `None`.
pub(crate) fn offset_within(
    idx: &[usize],
    offset: &[usize],
    extent: &[usize],
) -> Option<Vec<usize>> {
    idx.iter()
        .zip(offset)
        .zip(extent)
        .map(|((&i, &o), &n)| i.checked_sub(o).filter(|&j| j < n))
        .collect()
}
