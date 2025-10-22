//! Operation builders for Tensor Compute dialect
//!
//! Provides high-level Rust API for constructing TC dialect operations.

use anyhow::Result;

#[cfg(feature = "mlir")]
use melior::{
    Context,
    ir::{
        Location, Module, Value, Identifier,
        attribute::StringAttribute,
        operation::{OperationBuilder, OperationLike},
        r#type::{RankedTensorType, Type},
    },
};

/// Builder for Tensor Compute dialect operations
#[cfg(feature = "mlir")]
pub struct TCBuilder<'c> {
    context: &'c Context,
    module: Module<'c>,
    location: Location<'c>,
}

#[cfg(not(feature = "mlir"))]
pub struct TCBuilder {
    _phantom: std::marker::PhantomData<()>,
}

#[cfg(feature = "mlir")]
impl<'c> TCBuilder<'c> {
    /// Create a new builder with a given context
    ///
    /// # Arguments
    ///
    /// * `context` - MLIR context to use for building operations
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use melior::Context;
    /// use tc_mlir::TCBuilder;
    ///
    /// let context = Context::new();
    /// let builder = TCBuilder::new(&context);
    /// ```
    pub fn new(context: &'c Context) -> Self {
        let location = Location::unknown(context);
        let module = Module::new(location);

        Self {
            context,
            module,
            location,
        }
    }

    /// Get the constructed module
    pub fn module(&self) -> &Module<'c> {
        &self.module
    }

    /// Get a mutable reference to the location
    pub fn location(&self) -> Location<'c> {
        self.location
    }

    /// Create a ranked tensor type
    fn create_tensor_type(&self, shape: &[i64], element_type: Type<'c>) -> Type<'c> {
        RankedTensorType::new(
            &shape.iter().map(|&d| d as u64).collect::<Vec<_>>(),
            element_type,
            None
        ).into()
    }

    /// Build a tensor contraction operation
    ///
    /// # Arguments
    ///
    /// * `lhs` - Left operand value
    /// * `rhs` - Right operand value
    /// * `result_type` - Expected result tensor type
    /// * `indices` - Einsum notation string (e.g., "ij,jk->ik")
    ///
    /// # Returns
    ///
    /// MLIR Value representing the result of the contraction
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use melior::{Context, ir::{Value, r#type::Type}};
    /// use tc_mlir::TCBuilder;
    ///
    /// let context = Context::new();
    /// let builder = TCBuilder::new(&context);
    /// let f64_type = Type::float64(&context);
    /// let result_type = builder.create_tensor_type(&[10, 30], f64_type);
    ///
    /// // lhs and rhs are MLIR Values
    /// let result = builder.contract(lhs, rhs, result_type, "ij,jk->ik")?;
    /// ```
    pub fn contract(
        &self,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        result_type: Type<'c>,
        indices: &str,
    ) -> Result<Value<'c, 'c>> {
        let indices_attr = StringAttribute::new(self.context, indices);
        let indices_id = Identifier::new(self.context, "indices");

        let operation = OperationBuilder::new("tn.contract", self.location)
            .add_operands(&[lhs, rhs])
            .add_attributes(&[(indices_id, indices_attr.into())])
            .add_results(&[result_type])
            .build()?;

        Ok(operation.result(0)?.into())
    }

    // NOTE: build_contract_from_einsum() has been moved to the main ariadnetor crate
    // to avoid circular dependency. This crate (dialect) should not depend on
    // EinsumExpr which is defined in the main crate (DSL layer).

    /// Build an SVD operation
    ///
    /// # Arguments
    ///
    /// * `input` - Input matrix value
    /// * `u_type` - Type for U matrix result
    /// * `s_type` - Type for S vector result
    /// * `v_type` - Type for V matrix result
    /// * `max_chi` - Optional maximum bond dimension
    /// * `threshold` - Optional truncation threshold
    ///
    /// # Returns
    ///
    /// Tuple of (U, S, V) MLIR values
    pub fn svd(
        &self,
        input: Value<'c, '_>,
        u_type: Type<'c>,
        s_type: Type<'c>,
        v_type: Type<'c>,
        max_chi: Option<i64>,
        threshold: Option<f64>,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>, Value<'c, 'c>)> {
        use melior::ir::attribute::{IntegerAttribute, FloatAttribute};
        use melior::ir::r#type::IntegerType;

        let mut builder = OperationBuilder::new("tn.svd", self.location)
            .add_operands(&[input])
            .add_results(&[u_type, s_type, v_type]);

        // Add optional attributes
        if let Some(max_chi_val) = max_chi {
            let i64_type = IntegerType::new(self.context, 64);
            let max_chi_attr = IntegerAttribute::new(i64_type.into(), max_chi_val);
            let max_chi_id = Identifier::new(self.context, "max_chi");
            builder = builder.add_attributes(&[(max_chi_id, max_chi_attr.into())]);
        }

        if let Some(threshold_val) = threshold {
            let f64_type = Type::float64(self.context);
            let threshold_attr = FloatAttribute::new(self.context, f64_type, threshold_val);
            let threshold_id = Identifier::new(self.context, "threshold");
            builder = builder.add_attributes(&[(threshold_id, threshold_attr.into())]);
        }

        let operation = builder.build()?;

        Ok((
            operation.result(0)?.into(),
            operation.result(1)?.into(),
            operation.result(2)?.into(),
        ))
    }

    /// Build a QR decomposition operation
    ///
    /// # Arguments
    ///
    /// * `input` - Input matrix value
    /// * `q_type` - Type for Q matrix result
    /// * `r_type` - Type for R matrix result
    ///
    /// # Returns
    ///
    /// Tuple of (Q, R) MLIR values
    pub fn qr(
        &self,
        input: Value<'c, '_>,
        q_type: Type<'c>,
        r_type: Type<'c>,
    ) -> Result<(Value<'c, 'c>, Value<'c, 'c>)> {
        let operation = OperationBuilder::new("tn.qr", self.location)
            .add_operands(&[input])
            .add_results(&[q_type, r_type])
            .build()?;

        Ok((
            operation.result(0)?.into(),
            operation.result(1)?.into(),
        ))
    }

    /// Build a transpose operation
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor value
    /// * `result_type` - Result tensor type
    /// * `permutation` - Dimension permutation (e.g., [1, 0] for matrix transpose)
    ///
    /// # Returns
    ///
    /// MLIR Value representing the transposed tensor
    pub fn transpose(
        &self,
        input: Value<'c, '_>,
        result_type: Type<'c>,
        permutation: &[i64],
    ) -> Result<Value<'c, 'c>> {
        use melior::ir::attribute::DenseI64ArrayAttribute;

        let perm_attr = DenseI64ArrayAttribute::new(self.context, permutation);
        let perm_id = Identifier::new(self.context, "permutation");

        let operation = OperationBuilder::new("tn.transpose", self.location)
            .add_operands(&[input])
            .add_attributes(&[(perm_id, perm_attr.into())])
            .add_results(&[result_type])
            .build()?;

        Ok(operation.result(0)?.into())
    }

    /// Build a reshape operation
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor value
    /// * `result_type` - Result tensor type with new shape
    ///
    /// # Returns
    ///
    /// MLIR Value representing the reshaped tensor
    pub fn reshape(
        &self,
        input: Value<'c, '_>,
        result_type: Type<'c>,
    ) -> Result<Value<'c, 'c>> {
        let operation = OperationBuilder::new("tn.reshape", self.location)
            .add_operands(&[input])
            .add_results(&[result_type])
            .build()?;

        Ok(operation.result(0)?.into())
    }

    /// Build a truncate operation
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor value
    /// * `result_type` - Result tensor type
    /// * `max_chi` - Optional maximum bond dimension
    /// * `threshold` - Optional truncation threshold
    ///
    /// # Returns
    ///
    /// MLIR Value representing the truncated tensor
    pub fn truncate(
        &self,
        input: Value<'c, '_>,
        result_type: Type<'c>,
        max_chi: Option<i64>,
        threshold: Option<f64>,
    ) -> Result<Value<'c, 'c>> {
        use melior::ir::attribute::{IntegerAttribute, FloatAttribute};
        use melior::ir::r#type::IntegerType;

        let mut builder = OperationBuilder::new("tn.truncate", self.location)
            .add_operands(&[input])
            .add_results(&[result_type]);

        // Add optional attributes
        if let Some(max_chi_val) = max_chi {
            let i64_type = IntegerType::new(self.context, 64);
            let max_chi_attr = IntegerAttribute::new(i64_type.into(), max_chi_val);
            let max_chi_id = Identifier::new(self.context, "max_chi");
            builder = builder.add_attributes(&[(max_chi_id, max_chi_attr.into())]);
        }

        if let Some(threshold_val) = threshold {
            let f64_type = Type::float64(self.context);
            let threshold_attr = FloatAttribute::new(self.context, f64_type, threshold_val);
            let threshold_id = Identifier::new(self.context, "threshold");
            builder = builder.add_attributes(&[(threshold_id, threshold_attr.into())]);
        }

        let operation = builder.build()?;

        Ok(operation.result(0)?.into())
    }
}

// Non-mlir stub implementation
#[cfg(not(feature = "mlir"))]
impl TCBuilder {
    pub fn new() -> Result<Self> {
        anyhow::bail!("TCBuilder requires 'mlir' feature to be enabled")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "mlir")]
    fn test_builder_creation() {
        use melior::Context;

        let context = Context::new();
        let builder = TCBuilder::new(&context);

        // Verify builder was created
        assert!(builder.module().as_operation().verify());
    }

    #[test]
    #[cfg(not(feature = "mlir"))]
    fn test_builder_requires_feature() {
        let result = TCBuilder::new();
        assert!(result.is_err());
    }
}
