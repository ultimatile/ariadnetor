//! Einstein summation notation parser and contraction plan

use std::collections::{HashMap, HashSet};

/// Parsed einsum expression with N inputs (indices as ASCII codes)
///
/// Supports 1 to N input tensors. Output indices can be explicit (`->out`)
/// or implicitly inferred (free indices sorted alphabetically).
///
/// # Examples
///
/// ```
/// use arnet_core::EinsumExpr;
///
/// // Matrix multiplication
/// let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
/// assert_eq!(expr.num_inputs(), 2);
/// assert!(expr.is_matrix_multiply());
/// assert_eq!(expr.infer_output_shape(&[&[10, 20], &[20, 30]]).unwrap(), vec![10, 30]);
///
/// // Higher-dimensional contraction (not a plain matmul)
/// let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
/// assert_eq!(expr.out_indices(), &[b'i', b'l']);
/// assert_eq!(expr.contracted_indices(), vec![b'j', b'k']);
/// assert!(!expr.is_matrix_multiply());
///
/// // Element-wise: every index appears in the output, nothing is contracted
/// let expr = EinsumExpr::parse("ij,ij->ij").unwrap();
/// assert!(expr.contracted_indices().is_empty());
///
/// // Implicit output inference
/// let expr = EinsumExpr::parse("ij,jk").unwrap();
/// assert_eq!(expr.out_indices(), &[b'i', b'k']);
///
/// // Single tensor trace
/// let expr = EinsumExpr::parse("ii->").unwrap();
/// assert_eq!(expr.num_inputs(), 1);
///
/// // Errors: an output index absent from every input, or a non-alphabetic index
/// assert!(EinsumExpr::parse("ij,jk->im").is_err());
/// assert!(EinsumExpr::parse("i1,jk->ik").is_err());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EinsumExpr {
    inputs: Vec<Vec<u8>>,
    out_indices: Vec<u8>,
}

impl EinsumExpr {
    /// Parse an einsum expression from string notation.
    ///
    /// When `->` is present, output indices are explicit.
    /// When `->` is omitted, output is inferred as free indices (appearing
    /// exactly once across all inputs) sorted alphabetically.
    pub fn parse(notation: &str) -> Result<Self, String> {
        let notation: String = notation.chars().filter(|c| !c.is_whitespace()).collect();

        let (inputs_str, out_str) = if let Some((inp, out)) = notation.split_once("->") {
            (inp, Some(out))
        } else {
            (notation.as_str(), None)
        };

        let input_parts: Vec<&str> = inputs_str.split(',').collect();
        if input_parts.is_empty() {
            return Err("No input tensors specified".to_string());
        }

        let inputs: Vec<Vec<u8>> = input_parts
            .iter()
            .map(|s| Self::parse_indices(s))
            .collect::<Result<_, _>>()?;

        let out_indices = if let Some(out) = out_str {
            Self::parse_indices(out)?
        } else {
            Self::infer_output(&inputs)
        };

        let expr = Self {
            inputs,
            out_indices,
        };
        expr.validate()?;
        Ok(expr)
    }

    /// Infer output indices when `->` is omitted.
    ///
    /// Free indices (appearing exactly once across all inputs) sorted alphabetically.
    fn infer_output(inputs: &[Vec<u8>]) -> Vec<u8> {
        let mut counts: HashMap<u8, usize> = HashMap::new();
        for input in inputs {
            for &idx in input {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }
        let mut free: Vec<u8> = counts
            .into_iter()
            .filter(|&(_, count)| count == 1)
            .map(|(idx, _)| idx)
            .collect();
        free.sort();
        free
    }

    fn parse_indices(s: &str) -> Result<Vec<u8>, String> {
        s.chars()
            .map(|c| {
                if c.is_ascii_alphabetic() {
                    Ok(c as u8)
                } else {
                    Err(format!("Invalid index '{}': must be A-Z or a-z", c))
                }
            })
            .collect()
    }

    /// Validate the einsum expression.
    ///
    /// Checks that all output indices appear in at least one input tensor.
    pub fn validate(&self) -> Result<(), String> {
        let mut input_indices = HashSet::new();
        for input in &self.inputs {
            for &idx in input {
                input_indices.insert(idx);
            }
        }

        for &idx in &self.out_indices {
            if !input_indices.contains(&idx) {
                return Err(format!(
                    "Output index '{}' does not appear in any input tensor",
                    idx as char
                ));
            }
        }

        Ok(())
    }

    /// Get all input index lists
    pub fn inputs(&self) -> &[Vec<u8>] {
        &self.inputs
    }

    /// Get output indices
    pub fn out_indices(&self) -> &[u8] {
        &self.out_indices
    }

    /// Number of input tensors
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Convenience accessor for the first input's indices.
    ///
    /// # Panics
    ///
    /// Panics if the expression has no inputs.
    pub fn lhs_indices(&self) -> &[u8] {
        &self.inputs[0]
    }

    /// Convenience accessor for the second input's indices.
    ///
    /// # Panics
    ///
    /// Panics if the expression has fewer than 2 inputs.
    pub fn rhs_indices(&self) -> &[u8] {
        &self.inputs[1]
    }

    /// Get all unique indices across inputs and output
    pub fn all_indices(&self) -> HashSet<u8> {
        let mut indices = HashSet::new();
        for input in &self.inputs {
            indices.extend(input);
        }
        indices.extend(&self.out_indices);
        indices
    }

    /// Get contracted indices (appear in inputs but not in output),
    /// preserving the order of first appearance across inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arnet_core::EinsumExpr;
    /// let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
    /// assert_eq!(expr.contracted_indices(), vec![b'j', b'k']);
    /// ```
    pub fn contracted_indices(&self) -> Vec<u8> {
        let output_set: HashSet<u8> = self.out_indices.iter().copied().collect();
        let mut contracted = Vec::new();
        let mut seen = HashSet::new();

        for input in &self.inputs {
            for &idx in input {
                if !output_set.contains(&idx) && seen.insert(idx) {
                    contracted.push(idx);
                }
            }
        }

        contracted
    }

    /// Check if this is a matrix multiplication pattern:
    /// 2 inputs, 3 unique indices, each input has 2 indices, output has 2 indices,
    /// exactly 1 contracted index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arnet_core::EinsumExpr;
    /// assert!(EinsumExpr::parse("ij,jk->ik").unwrap().is_matrix_multiply());
    /// assert!(!EinsumExpr::parse("ijk,jkl->il").unwrap().is_matrix_multiply());
    /// ```
    pub fn is_matrix_multiply(&self) -> bool {
        if self.inputs.len() != 2 {
            return false;
        }

        let mut all_indices: HashSet<u8> = HashSet::new();
        for input in &self.inputs {
            all_indices.extend(input);
        }

        if all_indices.len() != 3 {
            return false;
        }

        if self.inputs[0].len() != 2 || self.inputs[1].len() != 2 {
            return false;
        }

        if self.out_indices.len() != 2 {
            return false;
        }

        self.contracted_indices().len() == 1
    }

    /// Infer the output tensor shape from input shapes.
    ///
    /// The number of shapes must match `num_inputs()`, and each shape's rank
    /// must match its corresponding input index count. Shared indices must
    /// have matching dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arnet_core::EinsumExpr;
    /// let expr = EinsumExpr::parse("ij,jk->ik").unwrap();
    /// assert_eq!(expr.infer_output_shape(&[&[10, 20], &[20, 30]]).unwrap(), vec![10, 30]);
    /// ```
    pub fn infer_output_shape(&self, shapes: &[&[usize]]) -> Result<Vec<usize>, String> {
        if shapes.len() != self.inputs.len() {
            return Err(format!(
                "Expected {} input shapes, got {}",
                self.inputs.len(),
                shapes.len()
            ));
        }

        for (i, (input, shape)) in self.inputs.iter().zip(shapes.iter()).enumerate() {
            if input.len() != shape.len() {
                return Err(format!(
                    "Input {} shape rank {} does not match index count {}",
                    i,
                    shape.len(),
                    input.len()
                ));
            }
        }

        // Build index → dimension mapping
        let mut index_dims: HashMap<u8, usize> = HashMap::new();

        for (input, shape) in self.inputs.iter().zip(shapes.iter()) {
            for (j, &idx) in input.iter().enumerate() {
                let dim = shape[j];
                if let Some(&existing_dim) = index_dims.get(&idx) {
                    if existing_dim != dim {
                        return Err(format!(
                            "Dimension mismatch for index '{}': found {} and {}",
                            idx as char, existing_dim, dim
                        ));
                    }
                } else {
                    index_dims.insert(idx, dim);
                }
            }
        }

        let mut output_shape = Vec::new();
        for &idx in &self.out_indices {
            let dim = index_dims.get(&idx).ok_or_else(|| {
                format!("Output index '{}' not found in input tensors", idx as char)
            })?;
            output_shape.push(*dim);
        }

        Ok(output_shape)
    }
}

/// Contraction plan identifying batch, contracted, and free indices for a 2-input einsum.
///
/// - **batch**: indices in both inputs AND in output (iterated over, not contracted)
/// - **contracted**: indices in both inputs but NOT in output (summed over)
/// - **free_lhs**: indices only in lhs and output (not in rhs)
/// - **free_rhs**: indices only in rhs and output (not in lhs)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractionPlan {
    pub batch: Vec<u8>,
    pub contracted: Vec<u8>,
    pub free_lhs: Vec<u8>,
    pub free_rhs: Vec<u8>,
}

impl ContractionPlan {
    pub fn from_expr(expr: &EinsumExpr) -> Self {
        let lhs = expr.lhs_indices();
        let rhs = expr.rhs_indices();
        let out = expr.out_indices();

        let lhs_set: HashSet<u8> = lhs.iter().copied().collect();
        let rhs_set: HashSet<u8> = rhs.iter().copied().collect();
        let out_set: HashSet<u8> = out.iter().copied().collect();

        // Contracted: in both lhs and rhs, not in output (preserve LHS order)
        let contracted: Vec<u8> = lhs
            .iter()
            .filter(|idx| rhs_set.contains(idx) && !out_set.contains(idx))
            .copied()
            .collect();

        // Batch: in both lhs and rhs AND in output (output order)
        let batch: Vec<u8> = out
            .iter()
            .filter(|idx| lhs_set.contains(idx) && rhs_set.contains(idx))
            .copied()
            .collect();

        let batch_set: HashSet<u8> = batch.iter().copied().collect();

        // Free lhs: in output and lhs, not in rhs (excludes batch)
        let free_lhs: Vec<u8> = out
            .iter()
            .filter(|idx| lhs_set.contains(idx) && !batch_set.contains(idx))
            .copied()
            .collect();

        // Free rhs: in output and rhs, not in lhs (excludes batch)
        let free_rhs: Vec<u8> = out
            .iter()
            .filter(|idx| rhs_set.contains(idx) && !batch_set.contains(idx))
            .copied()
            .collect();

        Self {
            batch,
            contracted,
            free_lhs,
            free_rhs,
        }
    }

    /// Compute LHS permutation to [batch, free_lhs, contracted] order.
    pub fn lhs_permutation(&self, lhs_indices: &[u8], rhs_indices: &[u8]) -> Option<Vec<usize>> {
        let contracted_set: HashSet<u8> = self.contracted.iter().copied().collect();
        let mut target = self.batch.clone();
        target.extend(&self.free_lhs);
        for &idx in rhs_indices {
            if contracted_set.contains(&idx) && !target.contains(&idx) {
                target.push(idx);
            }
        }
        compute_permutation(lhs_indices, &target)
    }

    /// Compute RHS permutation to [batch, contracted, free_rhs] order.
    pub fn rhs_permutation(&self, rhs_indices: &[u8]) -> Option<Vec<usize>> {
        let contracted_set: HashSet<u8> = self.contracted.iter().copied().collect();
        let mut target = self.batch.clone();
        for &idx in rhs_indices {
            if contracted_set.contains(&idx) && !target.contains(&idx) {
                target.push(idx);
            }
        }
        target.extend(&self.free_rhs);
        compute_permutation(rhs_indices, &target)
    }
}

/// Compute permutation from current to target order
pub fn compute_permutation(current: &[u8], target: &[u8]) -> Option<Vec<usize>> {
    assert_eq!(current.len(), target.len());
    let perm: Vec<usize> = target
        .iter()
        .map(|&idx| {
            current
                .iter()
                .position(|&x| x == idx)
                .expect("Index not found")
        })
        .collect();
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        None
    } else {
        Some(perm)
    }
}

#[cfg(test)]
mod tests;
