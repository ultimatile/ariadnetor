//===- TNTypes.td - Tensor Network Dialect Types ----------*- tablegen -*-===//
//
// Tensor Network Compute Dialect - Type definitions
//
//===----------------------------------------------------------------------===//

#ifndef TN_TYPES
#define TN_TYPES

include "tn-compute/Dialect/IR/TNBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// TN Type Definitions
//===----------------------------------------------------------------------===//

class TN_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TN_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Scalar Types
//===----------------------------------------------------------------------===//

// Supported scalar element types
def TN_Float : AnyTypeOf<[F16, F32, F64, BF16], "floating-point">;
def TN_Int : AnyTypeOf<[I8, I16, I32, I64], "integer">;
def TN_Scalar : AnyTypeOf<[TN_Float, TN_Int], "scalar">;

//===----------------------------------------------------------------------===//
// Tensor Type Constraints
//===----------------------------------------------------------------------===//

// Ranked tensor with TN scalar elements (primary tensor type)
// Future: define custom TN_TensorType with sparse/distributed metadata
def TN_Tensor : RankedTensorOf<[TN_Scalar]>;

// Float-only tensors (for operations like SVD)
def TN_FloatTensor : RankedTensorOf<[TN_Float]>;

// Specific dimension constraints
def TN_Vector : 1DTensorOf<[TN_Scalar]>;     // 1D tensor (vector)
def TN_Matrix : 2DTensorOf<[TN_Scalar]>;     // 2D tensor (matrix)

//===----------------------------------------------------------------------===//
// Future: Custom Tensor Type with Metadata
//===----------------------------------------------------------------------===//

// TODO: Define custom TN_TensorType with:
// - Sparse encoding (COO, CSR, etc.)
// - Distribution strategy (Block, Cyclic, etc.)
//
// def TN_TensorType : TN_Type<"Tensor", "tensor"> {
//   let summary = "Tensor Network tensor type";
//   let description = [{
//     A ranked tensor type with optional sparse encoding and
//     distribution metadata for distributed execution.
//   }];
//
//   let parameters = (ins
//     ArrayRefParameter<"int64_t">:$shape,
//     "Type":$elementType,
//     OptionalParameter<"Encoding">:$encoding,
//     OptionalParameter<"Distribution">:$distribution
//   );
// }

#endif // TN_TYPES
