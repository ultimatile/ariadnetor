//! Reshape, contiguity conversion, element-wise, and arithmetic operations.

use num_traits::Zero;
use std::ops::{Add, Mul};
use std::sync::Arc;

use super::{Dense, MemoryOrder, column_major_strides, row_major_strides};

impl<T> Dense<T>
where
    T: Clone,
{
    // ========================================================================
    // Reshape
    // ========================================================================

    /// Reshape the tensor to a new shape.
    ///
    /// Zero-copy if strides are compatible with the new shape.
    /// Otherwise, copies to contiguous layout first.
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

        if let Some(view) = self.reshape_view(new_shape.clone()) {
            return view;
        }

        // Non-contiguous: copy to contiguous first, then reshape
        let contiguous = self.to_contiguous(MemoryOrder::RowMajor);
        contiguous
            .reshape_view(new_shape)
            .expect("reshape_view failed on contiguous tensor")
    }

    /// Zero-copy reshape if strides are compatible with the new shape.
    ///
    /// Returns `None` if the tensor must be copied to support the new shape.
    pub fn reshape_view(&self, new_shape: Vec<usize>) -> Option<Self> {
        let new_total: usize = new_shape.iter().product();
        if self.len() != new_total {
            return None;
        }

        // For contiguous tensors, reshape is always zero-copy:
        // just compute new strides in the same memory order.
        if let Some(order) = self.contiguous_order() {
            let new_strides = match order {
                MemoryOrder::RowMajor => row_major_strides(&new_shape),
                MemoryOrder::ColumnMajor => column_major_strides(&new_shape),
            };
            return Some(Self {
                data: Arc::clone(&self.data),
                shape: new_shape,
                strides: new_strides,
                offset: self.offset,
                order,
            });
        }

        // Non-contiguous: cannot reshape without copying
        None
    }

    // ========================================================================
    // Contiguity conversion
    // ========================================================================

    /// Create a contiguous copy in the specified memory order.
    ///
    /// No-op (Arc clone) if already contiguous in the requested order.
    pub fn to_contiguous(&self, order: MemoryOrder) -> Self {
        let already_ok = match order {
            MemoryOrder::RowMajor => self.is_row_major() && self.offset == 0,
            MemoryOrder::ColumnMajor => self.is_column_major() && self.offset == 0,
        };

        if already_ok {
            return self.clone();
        }

        let total = self.len();
        let new_strides = match order {
            MemoryOrder::RowMajor => row_major_strides(&self.shape),
            MemoryOrder::ColumnMajor => column_major_strides(&self.shape),
        };

        // Iterate through all logical indices in the target order and copy
        let mut new_data = Vec::with_capacity(total);
        let rank = self.rank();
        let mut coords = vec![0usize; rank];

        // Iteration order depends on target layout
        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank).collect(),
            MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
        };

        for _ in 0..total {
            new_data.push(self.get(&coords));

            // Increment coordinates in the target order
            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                if coords[d] < self.shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Self::from_data_with_strides(new_data, self.shape.clone(), new_strides, 0, order)
    }

    // ========================================================================
    // Element-wise operations
    // ========================================================================

    /// Apply a function to each element, preserving the tensor's memory order.
    ///
    /// Iterates over contiguous data directly for efficiency.
    pub fn map<U, F>(&self, f: F) -> Dense<U>
    where
        F: Fn(&T) -> U,
        U: Clone + 'static,
    {
        let order = self.memory_order();
        let contiguous = self.to_contiguous(order);
        let result: Vec<U> = contiguous.data().iter().map(f).collect();
        Dense::from_data_with_order(result, self.shape().to_vec(), order)
    }

    /// Apply a function to each element in place (triggers CoW if shared).
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not contiguous.
    pub fn map_mut<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
    {
        assert!(
            self.is_contiguous(),
            "map_mut() requires contiguous tensor; \
             call to_contiguous() first"
        );
        let data = self.data_mut();
        for x in data.iter_mut() {
            *x = f(x);
        }
    }

    /// Apply a function with multi-dimensional coordinates to each element,
    /// preserving the tensor's memory order.
    pub fn map_with_index<U, F>(&self, f: F) -> Dense<U>
    where
        F: Fn(&[usize], &T) -> U,
        U: Clone + 'static,
    {
        let shape = self.shape();
        let rank = shape.len();
        let total = self.len();
        let order = self.memory_order();
        let mut coords = vec![0usize; rank];
        let mut result = Vec::with_capacity(total);

        let axis_order: Vec<usize> = match order {
            MemoryOrder::RowMajor => (0..rank).collect(),
            MemoryOrder::ColumnMajor => (0..rank).rev().collect(),
        };

        for _ in 0..total {
            let val = self.get(&coords);
            result.push(f(&coords, &val));

            for &d in axis_order.iter().rev() {
                coords[d] += 1;
                if coords[d] < shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Dense::from_data_with_order(result, shape.to_vec(), order)
    }

    // ========================================================================
    // Arithmetic operations
    // ========================================================================

    /// Scale all elements by a scalar factor (in-place).
    ///
    /// Preserves the tensor's memory order.
    pub fn scale<S>(&mut self, factor: S)
    where
        T: Mul<S, Output = T>,
        S: Clone,
    {
        *self = self.to_contiguous(self.memory_order());
        let data = self.data_mut();
        for elem in data.iter_mut() {
            *elem = elem.clone() * factor.clone();
        }
    }

    /// Scale all elements and return a new tensor (out-of-place).
    pub fn scaled(&self, factor: T) -> Self
    where
        T: Mul<Output = T>,
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
    /// Output memory order matches the first tensor's order.
    ///
    /// # Errors
    ///
    /// Returns an error if tensors have different shapes, the list is empty,
    /// or tensors and coefficients have different lengths.
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
        for t in &tensors[1..] {
            if t.shape() != shape {
                return Err("All tensors must have the same shape".to_string());
            }
        }
        let order = tensors[0].memory_order();
        let len = tensors[0].len();
        let mut result = vec![T::zero(); len];
        for (tensor, coef) in tensors.iter().zip(coefs) {
            let c = tensor.to_contiguous(order);
            for (r, val) in result.iter_mut().zip(c.data()) {
                *r = r.clone() + coef.clone() * val.clone();
            }
        }
        Ok(Dense::from_data_with_order(result, shape.to_vec(), order))
    }
}
