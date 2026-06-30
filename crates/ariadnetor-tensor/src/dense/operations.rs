//! Reshape, element-wise, and arithmetic operations for `DenseTensorData<T>`.

use num_traits::Zero;
use std::ops::{Add, Mul, MulAssign};

use crate::{DenseLayout, DenseTensorData, TensorData, TensorError};
use ariadnetor_core::MemoryOrder;

impl<T> DenseTensorData<T>
where
    T: Clone,
{
    /// Reshape the tensor to a new shape (zero-copy: shares the
    /// underlying storage Arc).
    ///
    /// The flat data is not rearranged — only the layout's shape
    /// changes. The output preserves `self.order()`. Reshape semantics
    /// depend on the order: adjacent-axis fusion is zero-copy under
    /// both row-major and column-major for contiguous tensors, but
    /// non-adjacent fusion produces a different logical mapping under
    /// each order.
    ///
    /// # Panics
    ///
    /// Panics if the new shape has a different total number of
    /// elements.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_total: usize = new_shape.iter().product();
        assert_eq!(
            self.len(),
            new_total,
            "reshape: total elements must match ({} vs {new_total})",
            self.len()
        );
        let storage = self.storage().clone();
        let layout = DenseLayout::new(new_shape, self.order());
        TensorData::new(storage, layout)
    }

    /// Apply a function to each element.
    ///
    /// Iterates flat data directly. The result preserves
    /// `self.order()`.
    pub fn map<U, F>(&self, f: F) -> DenseTensorData<U>
    where
        F: Fn(&T) -> U,
        U: Clone + 'static,
    {
        let result: Vec<U> = self.storage().data().iter().map(f).collect();
        DenseTensorData::<U>::from_raw_parts(result, self.shape().to_vec(), self.order())
    }

    /// Apply a function with multi-dimensional coordinates to each
    /// element.
    ///
    /// Iterates coordinates in `self.order()` while reading storage
    /// linearly, so the coordinate-to-value mapping always matches
    /// the storage's layout. The output preserves `self.order()`.
    pub fn map_with_index<U, F>(&self, f: F) -> DenseTensorData<U>
    where
        F: Fn(&[usize], &T) -> U,
        U: Clone + 'static,
    {
        let order = self.order();
        let shape = self.shape();
        let rank = shape.len();
        let total = self.len();
        let raw = self.storage().data();
        let mut coords = vec![0usize; rank];
        let mut result = Vec::with_capacity(total);

        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank).collect(),
            MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
        };

        for val in raw.iter().take(total) {
            result.push(f(&coords, val));
            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                if coords[d] < shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        DenseTensorData::<U>::from_raw_parts(result, shape.to_vec(), order)
    }

    /// Scale all elements and return a new tensor (out-of-place).
    pub fn scaled<S>(&self, factor: S) -> Self
    where
        T: Mul<S, Output = T>,
        S: Clone,
    {
        let mut result = self.clone();
        result.storage_mut().scale(factor);
        result
    }
}

// ============================================================================
// Scalar-multiplication operators on DenseTensorData<T>
// ============================================================================
//
// Convenience aliases for `scale` / `scaled`, restricted to a same-type
// factor (`S = T`). Cross-type factors (e.g. scaling a complex tensor by
// a real) cannot be expressed through a single `Mul` impl without
// conflicting coherence, so those callers keep using the named methods.

impl<T> Mul<T> for DenseTensorData<T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = DenseTensorData<T>;

    /// Scale by `rhs`, consuming `self`. Reuses the owned buffer in
    /// place (no extra allocation when the storage is uniquely owned;
    /// a buffer still shared via copy-on-write is cloned first).
    fn mul(mut self, rhs: T) -> Self::Output {
        self.scale(rhs);
        self
    }
}

impl<T> Mul<T> for &DenseTensorData<T>
where
    T: Clone + Mul<Output = T>,
{
    type Output = DenseTensorData<T>;

    /// Scale by `rhs`, leaving `self` untouched (out-of-place).
    fn mul(self, rhs: T) -> Self::Output {
        self.scaled(rhs)
    }
}

impl<T> MulAssign<T> for DenseTensorData<T>
where
    T: Clone + Mul<Output = T>,
{
    /// Scale every element by `rhs` in place.
    fn mul_assign(&mut self, rhs: T) {
        self.scale(rhs);
    }
}

// ============================================================================
// Multi-tensor arithmetic on DenseTensorData<T>
// ============================================================================

impl<T> DenseTensorData<T>
where
    T: Clone,
{
    /// Add all tensors (coefficients all = 1).
    pub fn add_all(tensors: &[&DenseTensorData<T>]) -> Result<DenseTensorData<T>, TensorError>
    where
        T: Zero + num_traits::One + Add<Output = T> + Mul<Output = T>,
    {
        let coefs = vec![T::one(); tensors.len()];
        Self::linear_combine(tensors, &coefs)
    }

    /// Linear combination: Σ coefs\[i\] * tensors\[i\].
    ///
    /// All input tensors must share the same `order()`; the result
    /// preserves that order.
    ///
    /// # Errors
    ///
    /// Returns an error if tensors have different shapes, different
    /// orders, the list is empty, or tensors and coefficients have
    /// different lengths.
    pub fn linear_combine(
        tensors: &[&DenseTensorData<T>],
        coefs: &[T],
    ) -> Result<DenseTensorData<T>, TensorError>
    where
        T: Zero + Add<Output = T> + Mul<Output = T>,
    {
        if tensors.is_empty() {
            return Err(TensorError::InvalidArgument(
                "Cannot combine empty tensor list".to_string(),
            ));
        }
        if tensors.len() != coefs.len() {
            return Err(TensorError::InvalidArgument(format!(
                "Mismatched lengths: {} tensors vs {} coefficients",
                tensors.len(),
                coefs.len()
            )));
        }
        let shape = tensors[0].shape();
        let order = tensors[0].order();
        for t in &tensors[1..] {
            if t.shape() != shape {
                return Err(TensorError::InvalidArgument(
                    "All tensors must have the same shape".to_string(),
                ));
            }
            if t.order() != order {
                return Err(TensorError::InvalidArgument(format!(
                    "All tensors must have the same memory order; got {:?} and {:?}",
                    order,
                    t.order()
                )));
            }
        }
        let len = tensors[0].len();
        let mut result = vec![T::zero(); len];
        for (tensor, coef) in tensors.iter().zip(coefs) {
            for (r, val) in result.iter_mut().zip(tensor.storage().data()) {
                *r = r.clone() + coef.clone() * val.clone();
            }
        }
        Ok(DenseTensorData::from_raw_parts(
            result,
            shape.to_vec(),
            order,
        ))
    }
}
