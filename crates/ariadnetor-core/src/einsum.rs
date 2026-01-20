//! Einstein summation notation parser and contraction plan

use std::collections::HashSet;

/// Parsed einsum expression (indices as ASCII codes)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EinsumExpr {
    pub lhs_indices: Vec<u8>,
    pub rhs_indices: Vec<u8>,
    pub out_indices: Vec<u8>,
}

impl EinsumExpr {
    pub fn parse(notation: &str) -> Result<Self, String> {
        let notation: String = notation.chars().filter(|c| !c.is_whitespace()).collect();
        let parts: Vec<&str> = notation.split("->").collect();
        if parts.len() != 2 {
            return Err(format!("Invalid einsum: expected 'inputs->output', got '{}'", notation));
        }

        let inputs: Vec<&str> = parts[0].split(',').collect();
        let (lhs_indices, rhs_indices) = match inputs.len() {
            1 => (Self::parse_indices(inputs[0])?, Vec::new()),
            2 => (Self::parse_indices(inputs[0])?, Self::parse_indices(inputs[1])?),
            _ => return Err(format!("Only unary or binary operations supported, got {}", inputs.len())),
        };

        Ok(Self {
            lhs_indices,
            rhs_indices,
            out_indices: Self::parse_indices(parts[1])?,
        })
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

    pub fn all_indices(&self) -> HashSet<u8> {
        let mut indices = HashSet::new();
        indices.extend(&self.lhs_indices);
        indices.extend(&self.rhs_indices);
        indices.extend(&self.out_indices);
        indices
    }
}

/// Contraction plan identifying contracted and free indices
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractionPlan {
    pub contracted: Vec<u8>,
    pub free_lhs: Vec<u8>,
    pub free_rhs: Vec<u8>,
}

impl ContractionPlan {
    pub fn from_expr(expr: &EinsumExpr) -> Self {
        let rhs_set: HashSet<u8> = expr.rhs_indices.iter().copied().collect();
        let out_set: HashSet<u8> = expr.out_indices.iter().copied().collect();
        let lhs_set: HashSet<u8> = expr.lhs_indices.iter().copied().collect();

        // Contracted: in both lhs and rhs, not in output (preserve LHS order)
        let contracted: Vec<u8> = expr.lhs_indices.iter()
            .filter(|idx| rhs_set.contains(idx) && !out_set.contains(idx))
            .copied()
            .collect();

        // Free indices in output order
        let free_lhs: Vec<u8> = expr.out_indices.iter()
            .filter(|idx| lhs_set.contains(idx))
            .copied()
            .collect();
        let free_rhs: Vec<u8> = expr.out_indices.iter()
            .filter(|idx| rhs_set.contains(idx))
            .copied()
            .collect();

        Self { contracted, free_lhs, free_rhs }
    }

    pub fn lhs_permutation(&self, lhs_indices: &[u8], rhs_indices: &[u8]) -> Option<Vec<usize>> {
        let contracted_set: HashSet<u8> = self.contracted.iter().copied().collect();
        let mut target = self.free_lhs.clone();
        for &idx in rhs_indices {
            if contracted_set.contains(&idx) && !target.contains(&idx) {
                target.push(idx);
            }
        }
        compute_permutation(lhs_indices, &target)
    }

    pub fn rhs_permutation(&self, rhs_indices: &[u8]) -> Option<Vec<usize>> {
        let contracted_set: HashSet<u8> = self.contracted.iter().copied().collect();
        let mut target = Vec::new();
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
    let perm: Vec<usize> = target.iter()
        .map(|&idx| current.iter().position(|&x| x == idx).expect("Index not found"))
        .collect();
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        None
    } else {
        Some(perm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_einsum() {
        let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
        assert_eq!(expr.lhs_indices, vec![b'i', b'j', b'k']);
        assert_eq!(expr.rhs_indices, vec![b'j', b'k', b'l']);
        assert_eq!(expr.out_indices, vec![b'i', b'l']);
    }

    #[test]
    fn test_contraction_plan() {
        let expr = EinsumExpr::parse("ijk,jkl->il").unwrap();
        let plan = ContractionPlan::from_expr(&expr);
        assert_eq!(plan.contracted, vec![b'j', b'k']);
        assert_eq!(plan.free_lhs, vec![b'i']);
        assert_eq!(plan.free_rhs, vec![b'l']);
    }

    #[test]
    fn test_contraction_preserves_lhs_order() {
        let expr = EinsumExpr::parse("ikj,jkl->il").unwrap();
        let plan = ContractionPlan::from_expr(&expr);
        assert_eq!(plan.contracted, vec![b'k', b'j']); // LHS order: k before j
    }
}
