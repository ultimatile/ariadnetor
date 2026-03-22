//! Expression Compute Graph for lazy evaluation
//!
//! This module provides an expression graph that represents tensor computations
//! in a deferred manner. Operations are recorded as graph nodes and only executed
//! when `evaluate()` is called.
//!
//! # Design Note
//!
//! Named `ExpressionComputeGraph` (not just `ExpressionGraph`) to avoid confusion
//! with tensor network graphs, which represent the connectivity structure of tensors
//! in a quantum/tensor network.
//!
//! - **ExpressionComputeGraph**: Computation dependency graph (this module)
//! - **Tensor Network Graph**: Tensor connectivity structure (future: tn.graph IR)
//!
//! # Example
//!
//! ```rust,ignore
//! use arnet::{Tensor, ExpressionComputeGraph};
//!
//! let a = Tensor::new(vec![10, 20]);
//! let b = Tensor::new(vec![20, 30]);
//! let c = Tensor::new(vec![30, 40]);
//!
//! // Build expression graph (no computation yet)
//! let expr = ExpressionComputeGraph::from_tensor(a)
//!     .contract(&ExpressionComputeGraph::from_tensor(b), "ij,jk->ik")
//!     .contract(&ExpressionComputeGraph::from_tensor(c), "ik,kl->il");
//!
//! // Evaluate triggers compilation and execution
//! let result = expr.evaluate()?;
//! ```

use crate::Tensor;
use anyhow::{Result, bail};

/// Expression compute graph representing deferred tensor computations
///
/// This is the core structure for lazy evaluation. Operations are recorded
/// as graph nodes and compiled/executed only when `evaluate()` is called.
#[derive(Debug, Clone)]
pub struct ExpressionComputeGraph {
    root: ExprNode,
}

/// A node in the expression compute graph
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields will be read when evaluate() is implemented
enum ExprNode {
    /// Leaf node containing a concrete tensor
    Leaf(Tensor),

    /// Binary tensor contraction (Einstein summation)
    Contract {
        lhs: Box<ExprNode>,
        rhs: Box<ExprNode>,
        einsum: String,
    },

    /// Multi-dimensional axis permutation
    Permute {
        input: Box<ExprNode>,
        permutation: Vec<usize>,
    },

    /// Trace over bond pairs
    Trace {
        input: Box<ExprNode>,
        pairs: Vec<(usize, usize)>,
    },

    /// Linear combination: out = Σ coefs\[i\] × tensors\[i\]
    LinearCombine {
        tensors: Vec<ExprNode>,
        coefs: Vec<f64>,
    },

    /// Element-wise scaling: out = scalar × input
    Scale { input: Box<ExprNode>, factor: f64 },

    /// Element-wise addition: out = lhs + rhs
    Add {
        lhs: Box<ExprNode>,
        rhs: Box<ExprNode>,
    },
}

impl ExpressionComputeGraph {
    /// Create an expression graph from a concrete tensor (leaf node)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tensor = Tensor::new(vec![10, 20]);
    /// let expr = ExpressionComputeGraph::from_tensor(tensor);
    /// ```
    pub fn from_tensor(tensor: Tensor) -> Self {
        Self {
            root: ExprNode::Leaf(tensor),
        }
    }

    /// Contract this expression with another using Einstein summation
    ///
    /// Records a contraction operation in the graph. No computation is performed
    /// until `evaluate()` is called.
    ///
    /// # Arguments
    ///
    /// * `other` - Right-hand side expression
    /// * `einsum` - Einstein summation notation (e.g., "ij,jk->ik")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let a = ExpressionComputeGraph::from_tensor(tensor_a);
    /// let b = ExpressionComputeGraph::from_tensor(tensor_b);
    /// let c = a.contract(&b, "ij,jk->ik");
    /// ```
    pub fn contract(self, other: &Self, einsum: &str) -> Self {
        Self {
            root: ExprNode::Contract {
                lhs: Box::new(self.root),
                rhs: Box::new(other.root.clone()),
                einsum: einsum.to_string(),
            },
        }
    }

    /// Permute axes of the tensor
    ///
    /// Records a permutation operation. For example, `permutation = [1, 0, 2]`
    /// transposes the first two axes.
    ///
    /// # Arguments
    ///
    /// * `permutation` - New order of axes
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let expr = ExpressionComputeGraph::from_tensor(tensor);
    /// let transposed = expr.permute(vec![1, 0, 2]); // Transpose first two axes
    /// ```
    pub fn permute(self, permutation: Vec<usize>) -> Self {
        Self {
            root: ExprNode::Permute {
                input: Box::new(self.root),
                permutation,
            },
        }
    }

    /// Trace over bond pairs
    ///
    /// Contracts specified pairs of axes. For example, `pairs = [(1, 3), (2, 4)]`
    /// contracts axis 1 with 3 and axis 2 with 4.
    ///
    /// # Arguments
    ///
    /// * `pairs` - Pairs of axes to contract
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let expr = ExpressionComputeGraph::from_tensor(tensor);
    /// let traced = expr.trace(vec![(1, 3), (2, 4)]);
    /// ```
    pub fn trace(self, pairs: Vec<(usize, usize)>) -> Self {
        Self {
            root: ExprNode::Trace {
                input: Box::new(self.root),
                pairs,
            },
        }
    }

    /// Scale tensor by a scalar factor
    ///
    /// Records an element-wise scaling operation: out = factor × input
    ///
    /// # Arguments
    ///
    /// * `factor` - Scaling factor
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let expr = ExpressionComputeGraph::from_tensor(tensor);
    /// let scaled = expr.scale(2.0);
    /// ```
    pub fn scale(self, factor: f64) -> Self {
        Self {
            root: ExprNode::Scale {
                input: Box::new(self.root),
                factor,
            },
        }
    }

    /// Add two expressions element-wise
    ///
    /// Records an addition operation: out = lhs + rhs
    ///
    /// # Arguments
    ///
    /// * `other` - Right-hand side expression
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let a = ExpressionComputeGraph::from_tensor(tensor_a);
    /// let b = ExpressionComputeGraph::from_tensor(tensor_b);
    /// let sum = a.add(&b);
    /// ```
    pub fn add(self, other: &Self) -> Self {
        Self {
            root: ExprNode::Add {
                lhs: Box::new(self.root),
                rhs: Box::new(other.root.clone()),
            },
        }
    }

    /// Create a linear combination of multiple expressions
    ///
    /// Records a linear combination: out = Σ coefs\[i\] × tensors\[i\]
    ///
    /// # Arguments
    ///
    /// * `tensors` - List of expressions
    /// * `coefs` - Corresponding coefficients
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = ExpressionComputeGraph::linear_combine(
    ///     vec![expr_a, expr_b, expr_c],
    ///     vec![1.0, 2.0, 3.0],
    /// );
    /// ```
    pub fn linear_combine(tensors: Vec<Self>, coefs: Vec<f64>) -> Result<Self> {
        if tensors.len() != coefs.len() {
            bail!(
                "Number of tensors ({}) must match number of coefficients ({})",
                tensors.len(),
                coefs.len()
            );
        }

        let nodes: Vec<ExprNode> = tensors.into_iter().map(|t| t.root).collect();

        Ok(Self {
            root: ExprNode::LinearCombine {
                tensors: nodes,
                coefs,
            },
        })
    }

    /// Evaluate the expression graph
    ///
    /// This triggers the actual computation:
    /// 1. Convert expression graph to MLIR IR
    /// 2. Apply optimization passes
    /// 3. JIT compile to native code
    /// 4. Execute and return result
    ///
    /// # Returns
    ///
    /// The computed tensor result
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let expr = a.contract(&b, "ij,jk->ik").scale(2.0);
    /// let result = expr.evaluate()?;
    /// ```
    pub fn evaluate(self) -> Result<Tensor> {
        // TODO: Implement JIT compilation
        // 1. Convert expression graph to MLIR IR (using TCBuilder)
        // 2. Apply optimization passes
        // 3. JIT compile using ExecutionEngine
        // 4. Execute and return result
        unimplemented!("JIT compilation not yet implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NativeBackend;
    use arnet_tensor::{DenseTensor, MemoryOrder, TensorStorage};

    fn tensor_from_data(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        Tensor::with_backend(
            TensorStorage::Dense(DenseTensor::from_data_with_order(
                data,
                shape,
                MemoryOrder::RowMajor,
            )),
            NativeBackend::shared(),
        )
    }

    #[test]
    #[ignore] // TODO: Implement JIT compilation
    fn test_from_tensor() {
        let tensor = tensor_from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let expr = ExpressionComputeGraph::from_tensor(tensor.clone());

        let result = expr.evaluate().unwrap();
        assert_eq!(result.data().unwrap(), tensor.data().unwrap());
    }

    #[test]
    #[ignore] // TODO: Implement JIT compilation
    fn test_scale() {
        let tensor = tensor_from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let expr = ExpressionComputeGraph::from_tensor(tensor);

        let scaled = expr.scale(2.0);
        let result = scaled.evaluate().unwrap();

        assert_eq!(result.data().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    #[ignore] // TODO: Implement JIT compilation
    fn test_add() {
        let a = tensor_from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor_from_data(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

        let expr_a = ExpressionComputeGraph::from_tensor(a);
        let expr_b = ExpressionComputeGraph::from_tensor(b);

        let sum = expr_a.add(&expr_b);
        let result = sum.evaluate().unwrap();

        assert_eq!(result.data().unwrap(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    #[ignore] // TODO: Implement JIT compilation
    fn test_chained_operations() {
        let a = tensor_from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor_from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let expr = ExpressionComputeGraph::from_tensor(a)
            .scale(2.0)
            .add(&ExpressionComputeGraph::from_tensor(b).scale(3.0));

        let result = expr.evaluate().unwrap();

        // (a * 2) + (b * 3) = [2,4,6,8] + [15,18,21,24] = [17,22,27,32]
        assert_eq!(result.data().unwrap(), &[17.0, 22.0, 27.0, 32.0]);
    }

    #[test]
    #[ignore] // TODO: Implement JIT compilation
    fn test_linear_combine() {
        let a = tensor_from_data(vec![1.0, 2.0], vec![2]);
        let b = tensor_from_data(vec![3.0, 4.0], vec![2]);
        let c = tensor_from_data(vec![5.0, 6.0], vec![2]);

        let expr = ExpressionComputeGraph::linear_combine(
            vec![
                ExpressionComputeGraph::from_tensor(a),
                ExpressionComputeGraph::from_tensor(b),
                ExpressionComputeGraph::from_tensor(c),
            ],
            vec![1.0, 2.0, 3.0],
        )
        .unwrap();

        let result = expr.evaluate().unwrap();

        // 1*[1,2] + 2*[3,4] + 3*[5,6] = [1,2] + [6,8] + [15,18] = [22,28]
        assert_eq!(result.data().unwrap(), &[22.0, 28.0]);
    }

    #[test]
    #[ignore] // TODO: Implement JIT compilation
    fn test_contract() {
        let a = tensor_from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor_from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let expr_a = ExpressionComputeGraph::from_tensor(a);
        let expr_b = ExpressionComputeGraph::from_tensor(b);

        let expr = expr_a.contract(&expr_b, "ij,jk->ik");
        let result = expr.evaluate().unwrap();

        // Expected: matrix multiplication
        // [1 2] × [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        assert_eq!(result.data().unwrap(), &[19.0, 22.0, 43.0, 50.0]);
    }
}
