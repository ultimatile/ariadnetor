//! Convenience constructors and accessors for `DenseTensorData<T>`.

use std::sync::Arc;

use aligned_vec::{AVec, ConstAlign};
use arnet_core::backend::MemoryOrder;

use crate::{DenseLayout, DenseStorage, TensorData};

/// Backend-less Dense tensor bundle = `TensorData<DenseStorage<T>, DenseLayout>`.
pub type DenseTensorData<T = f64> = TensorData<DenseStorage<T>, DenseLayout>;

impl<T> DenseTensorData<T> {
    /// Construct from flat data, shape, and the memory order the
    /// data is laid out in.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal `shape.iter().product()`.
    pub fn from_raw_parts(data: Vec<T>, shape: Vec<usize>, order: MemoryOrder) -> Self
    where
        T: Clone,
    {
        let storage = DenseStorage::new(data);
        let layout = DenseLayout::new(shape, order);
        Self::new(storage, layout)
    }

    /// Reference to the flat data buffer.
    pub fn data(&self) -> &[T] {
        self.storage().data()
    }

    /// Logical shape.
    pub fn shape(&self) -> &[usize] {
        self.layout().shape()
    }

    /// Memory order the flat data is laid out in.
    pub fn order(&self) -> MemoryOrder {
        self.layout().order()
    }

    /// Get element at multi-dimensional indices.
    ///
    /// The flat index is computed using `self.order()`, so a
    /// `RowMajor`-tagged and a `ColumnMajor`-tagged tensor holding
    /// the same logical matrix in their respective layouts return
    /// the same value at the same `[i, j, ...]`.
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    pub fn get(&self, indices: &[usize]) -> T
    where
        T: Clone,
    {
        let shape = self.shape();
        assert_eq!(indices.len(), shape.len());
        for (axis, (&idx, &dim)) in indices.iter().zip(shape).enumerate() {
            assert!(
                idx < dim,
                "index {idx} out of bounds for axis {axis} with size {dim}"
            );
        }
        let idx = crate::reorder::flat_index(indices, shape, self.order());
        self.data()[idx].clone()
    }

    /// Rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.layout().rank()
    }

    /// Total number of logical elements (`shape().iter().product()`).
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Whether the tensor has zero logical elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Zero-filled tensor in `ColumnMajor` order. Uniform-value data
    /// is layout-invariant; backend-aware callers should construct
    /// directly via [`from_raw_parts`](Self::from_raw_parts) with
    /// `backend.preferred_order()` when matching is required.
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Clone + num_traits::Zero,
    {
        let total: usize = shape.iter().product();
        Self::from_raw_parts(vec![T::zero(); total], shape, MemoryOrder::ColumnMajor)
    }

    /// Ones-filled tensor in `ColumnMajor` order. See
    /// [`zeros`](Self::zeros) for the order convention.
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: Clone + num_traits::One + num_traits::Zero,
    {
        let total: usize = shape.iter().product();
        Self::from_raw_parts(vec![T::one(); total], shape, MemoryOrder::ColumnMajor)
    }

    /// Constant-filled tensor in `ColumnMajor` order.
    pub fn constant(shape: Vec<usize>, value: T) -> Self
    where
        T: Clone,
    {
        let total: usize = shape.iter().product();
        Self::from_raw_parts(vec![value; total], shape, MemoryOrder::ColumnMajor)
    }

    /// `n×n` identity matrix in `ColumnMajor` order. The identity
    /// matrix is symmetric, so the flat layout is the same under
    /// either memory order; only the layout's tag differs from
    /// `RowMajor`.
    pub fn eye(n: usize) -> Self
    where
        T: Clone + num_traits::Zero + num_traits::One,
    {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        Self::from_raw_parts(data, vec![n, n], MemoryOrder::ColumnMajor)
    }

    /// Random tensor with elements drawn from the standard
    /// distribution. Random-valued data is layout-invariant; the
    /// resulting tensor's order is `ColumnMajor`.
    pub fn random<R: rand::Rng>(shape: Vec<usize>, rng: &mut R) -> Self
    where
        T: Clone,
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let total: usize = shape.iter().product();
        let data: Vec<T> = (0..total).map(|_| rng.random()).collect();
        Self::from_raw_parts(data, shape, MemoryOrder::ColumnMajor)
    }

    /// Reshape the tensor to a new shape (zero-copy on storage).
    ///
    /// The flat data is not rearranged — only the layout changes.
    /// The output preserves `self.order()`. Reshape semantics depend
    /// on the order: adjacent-axis fusion is zero-copy under both
    /// row-major and column-major for contiguous tensors, but
    /// non-adjacent fusion produces a different logical mapping
    /// under each order.
    ///
    /// # Panics
    ///
    /// Panics if the new shape has a different total element count
    /// than the current shape.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self
    where
        T: Clone,
    {
        let new_total: usize = new_shape.iter().product();
        assert_eq!(
            self.len(),
            new_total,
            "reshape: total elements must match ({} vs {new_total})",
            self.len()
        );
        let storage = self.storage().clone();
        let layout = DenseLayout::new(new_shape, self.order());
        Self::new(storage, layout)
    }
}

impl<T> DenseTensorData<T>
where
    T: arnet_core::Scalar,
{
    /// Element-wise complex conjugate. Layout is preserved.
    pub fn conj(&self) -> Self {
        let new_data: AVec<T, ConstAlign<64>> =
            AVec::from_iter(64, self.data().iter().copied().map(|x| x.conj()));
        let storage = DenseStorage::from_arc(Arc::new(new_data));
        Self::new(storage, self.layout().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn zeros_populates_with_zero_in_column_major() {
        let t = DenseTensorData::<f64>::zeros(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.order(), MemoryOrder::ColumnMajor);
        assert_eq!(t.data(), &[0.0; 6]);
        assert_eq!(t.len(), 6);
        assert!(!t.is_empty());
    }

    #[test]
    fn ones_populates_with_one_in_column_major() {
        let t = DenseTensorData::<f64>::ones(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.order(), MemoryOrder::ColumnMajor);
        assert_eq!(t.data(), &[1.0; 6]);
    }

    #[test]
    fn constant_populates_with_value() {
        let t = DenseTensorData::<f64>::constant(vec![3, 2], 7.5);
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.data(), &[7.5; 6]);
    }

    #[test]
    fn eye_is_column_major_identity() {
        let t = DenseTensorData::<f64>::eye(3);
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(t.order(), MemoryOrder::ColumnMajor);
        // Column-major flat order: col 0 = [1,0,0], col 1 = [0,1,0], col 2 = [0,0,1].
        assert_eq!(t.data(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(t.get(&[i, j]), expected);
            }
        }
    }

    #[test]
    fn random_populates_shape_and_is_column_major() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let t = DenseTensorData::<f64>::random(vec![2, 4], &mut rng);
        assert_eq!(t.shape(), &[2, 4]);
        assert_eq!(t.order(), MemoryOrder::ColumnMajor);
        assert_eq!(t.data().len(), 8);
    }

    #[test]
    fn reshape_preserves_order_and_data_but_changes_shape() {
        let t = DenseTensorData::<f64>::from_raw_parts(
            (0..12).map(|i| i as f64).collect(),
            vec![3, 4],
            MemoryOrder::ColumnMajor,
        );
        let r = t.reshape(vec![2, 6]);
        assert_eq!(r.shape(), &[2, 6]);
        assert_eq!(r.order(), MemoryOrder::ColumnMajor);
        assert_eq!(r.data(), t.data());
    }

    #[test]
    #[should_panic(expected = "reshape: total elements must match")]
    fn reshape_panics_on_total_mismatch() {
        let t = DenseTensorData::<f64>::zeros(vec![2, 3]);
        let _ = t.reshape(vec![2, 4]);
    }

    #[test]
    fn conj_negates_imaginary_part_and_preserves_layout() {
        let data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, -4.0),
            Complex::new(0.0, 5.0),
            Complex::new(-1.0, 0.0),
        ];
        let t = DenseTensorData::<Complex<f64>>::from_raw_parts(
            data.clone(),
            vec![2, 2],
            MemoryOrder::ColumnMajor,
        );
        let c = t.conj();
        assert_eq!(c.shape(), t.shape());
        assert_eq!(c.order(), t.order());
        for (orig, conj) in data.iter().zip(c.data().iter()) {
            assert_eq!(conj.re, orig.re);
            assert_eq!(conj.im, -orig.im);
        }
    }

    #[test]
    fn from_raw_parts_panics_on_length_mismatch() {
        let result = std::panic::catch_unwind(|| {
            DenseTensorData::<f64>::from_raw_parts(
                vec![1.0, 2.0, 3.0],
                vec![2, 2],
                MemoryOrder::ColumnMajor,
            )
        });
        assert!(result.is_err());
    }

    #[test]
    fn get_reads_through_layout_order() {
        // Row-major-tagged 2×3 matrix `[[10, 20, 30], [40, 50, 60]]`.
        let t_rm = DenseTensorData::<f64>::from_raw_parts(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![2, 3],
            MemoryOrder::RowMajor,
        );
        assert_eq!(t_rm.get(&[0, 0]), 10.0);
        assert_eq!(t_rm.get(&[1, 2]), 60.0);

        // Same logical matrix stored column-major: col 0 = [10,40], col 1 = [20,50], col 2 = [30,60].
        let t_cm = DenseTensorData::<f64>::from_raw_parts(
            vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0],
            vec![2, 3],
            MemoryOrder::ColumnMajor,
        );
        assert_eq!(t_cm.get(&[0, 0]), 10.0);
        assert_eq!(t_cm.get(&[1, 2]), 60.0);
        // Same logical element via different storage layouts.
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(t_rm.get(&[i, j]), t_cm.get(&[i, j]));
            }
        }
    }
}
