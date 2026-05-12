//! Reshape, element-wise, and arithmetic operations.

use num_traits::Zero;
use std::ops::{Add, Mul};
use std::sync::Arc;

use super::Dense;
use arnet_core::MemoryOrder;

impl<T> Dense<T>
where
    T: Clone,
{
    // ========================================================================
    // Reshape
    // ========================================================================

    /// Reshape the tensor to a new shape (zero-copy).
    ///
    /// The flat data is not rearranged — only the shape changes.
    /// The output preserves `self.order()`. Reshape semantics depend on
    /// the order: adjacent-axis fusion is zero-copy under both row-major
    /// and column-major for contiguous tensors, but non-adjacent fusion
    /// produces a different logical mapping under each order.
    ///
    /// # Panics
    ///
    /// Panics if the new shape has a different total number of elements.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_total: usize = new_shape.iter().product();
        assert_eq!(
            self.len(),
            new_total,
            "reshape: total elements must match ({} vs {new_total})",
            self.len()
        );

        Self {
            data: Arc::clone(&self.data),
            shape: new_shape,
            order: self.order,
        }
    }

    // ========================================================================
    // Element-wise operations
    // ========================================================================

    /// Apply a function to each element.
    ///
    /// Iterates flat data directly. The result preserves `self.order()`.
    pub fn map<U, F>(&self, f: F) -> Dense<U>
    where
        F: Fn(&T) -> U,
        U: Clone + 'static,
    {
        let result: Vec<U> = self.data().iter().map(f).collect();
        Dense::new(result, self.shape().to_vec(), self.order())
    }

    /// Apply a function to each element in place (triggers CoW if shared).
    pub fn map_mut<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
    {
        let data = self.data_mut();
        for x in data.iter_mut() {
            *x = f(x);
        }
    }

    /// Apply a function with multi-dimensional coordinates to each element.
    ///
    /// Iterates coordinates in `self.order()` while reading the storage
    /// linearly, so the coordinate-to-value mapping always matches the
    /// storage's layout. The output preserves `self.order()`.
    pub fn map_with_index<U, F>(&self, f: F) -> Dense<U>
    where
        F: Fn(&[usize], &T) -> U,
        U: Clone + 'static,
    {
        let order = self.order();
        let shape = self.shape();
        let rank = shape.len();
        let total = self.len();
        let raw = self.data();
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

        Dense::new(result, shape.to_vec(), order)
    }

    // ========================================================================
    // Arithmetic operations
    // ========================================================================

    /// Scale all elements by a scalar factor (in-place).
    pub fn scale<S>(&mut self, factor: S)
    where
        T: Mul<S, Output = T>,
        S: Clone,
    {
        let data = self.data_mut();
        for elem in data.iter_mut() {
            *elem = elem.clone() * factor.clone();
        }
    }

    /// Scale all elements and return a new tensor (out-of-place).
    pub fn scaled<S>(&self, factor: S) -> Self
    where
        T: Mul<S, Output = T>,
        S: Clone,
    {
        let mut result = self.clone();
        result.scale(factor);
        result
    }

    /// Add all tensors (coefficients all = 1).
    pub fn add_all(tensors: &[&Dense<T>]) -> Result<Dense<T>, String>
    where
        T: Zero + num_traits::One + Add<Output = T> + Mul<Output = T>,
    {
        let coefs = vec![T::one(); tensors.len()];
        Self::linear_combine(tensors, &coefs)
    }

    /// Linear combination: Σ coefs\[i\] * tensors\[i\].
    ///
    /// All input tensors must share the same `order()`; the result
    /// preserves that order. Mixing tensors with different memory
    /// orders is rejected — convert via `reorder` upstream so
    /// element-wise summation aligns logical positions correctly.
    ///
    /// # Errors
    ///
    /// Returns an error if tensors have different shapes, different
    /// orders, the list is empty, or tensors and coefficients have
    /// different lengths.
    pub fn linear_combine(tensors: &[&Dense<T>], coefs: &[T]) -> Result<Dense<T>, String>
    where
        T: Zero + Add<Output = T> + Mul<Output = T>,
    {
        if tensors.is_empty() {
            return Err("Cannot combine empty tensor list".to_string());
        }
        if tensors.len() != coefs.len() {
            return Err(format!(
                "Mismatched lengths: {} tensors vs {} coefficients",
                tensors.len(),
                coefs.len()
            ));
        }
        let shape = tensors[0].shape();
        let order = tensors[0].order();
        for t in &tensors[1..] {
            if t.shape() != shape {
                return Err("All tensors must have the same shape".to_string());
            }
            if t.order() != order {
                return Err(format!(
                    "All tensors must have the same memory order; got {:?} and {:?}",
                    order,
                    t.order()
                ));
            }
        }
        let len = tensors[0].len();
        let mut result = vec![T::zero(); len];
        for (tensor, coef) in tensors.iter().zip(coefs) {
            for (r, val) in result.iter_mut().zip(tensor.data()) {
                *r = r.clone() + coef.clone() * val.clone();
            }
        }
        Ok(Dense::new(result, shape.to_vec(), order))
    }
}
