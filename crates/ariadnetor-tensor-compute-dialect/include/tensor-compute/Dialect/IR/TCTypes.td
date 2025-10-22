//===- TCTypes.td - Tensor Compute Dialect Types ----------*- tablegen -*-===//
//
// Tensor Compute Dialect - Type definitions
//
//===----------------------------------------------------------------------===//

#ifndef TC_TYPES
#define TC_TYPES

include "tensor-compute/Dialect/IR/TCBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// TC Type Definitions
//===----------------------------------------------------------------------===//

class TC_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TC_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Scalar Types
//===----------------------------------------------------------------------===//

// Supported scalar element types
def TC_Float : AnyTypeOf<[F16, F32, F64, BF16], "floating-point">;
def TC_Int : AnyTypeOf<[I8, I16, I32, I64], "integer">;
def TC_Scalar : AnyTypeOf<[TC_Float, TC_Int], "scalar">;

//===----------------------------------------------------------------------===//
// Tensor Type Constraints
//===----------------------------------------------------------------------===//

// Ranked tensor with TC scalar elements (primary tensor type)
// Future: define custom TC_TensorType with sparse/distributed metadata
def TC_Tensor : RankedTensorOf<[TC_Scalar]>;

// Float-only tensors (for operations like SVD)
def TC_FloatTensor : RankedTensorOf<[TC_Float]>;

// Specific dimension constraints
def TC_Vector : 1DTensorOf<[TC_Scalar]>;     // 1D tensor (vector)
def TC_Matrix : 2DTensorOf<[TC_Scalar]>;     // 2D tensor (matrix)

//===----------------------------------------------------------------------===//
// Future: Custom Tensor Type with Metadata
//===----------------------------------------------------------------------===//

// TODO: Define custom TC_TensorType with:
// - Sparse encoding (COO, CSR, etc.)
// - Distribution strategy (Block, Cyclic, etc.)
//
// def TC_TensorType : TC_Type<"Tensor", "tensor"> {
//   let summary = "Tensor Compute tensor type";
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

#endif // TC_TYPES
