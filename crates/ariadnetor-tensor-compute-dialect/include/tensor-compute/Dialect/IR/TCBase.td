//===- TCBase.td - Tensor Compute Dialect Base ------------*- tablegen -*-===//
//
// Tensor Compute Dialect - Base definitions
//
//===----------------------------------------------------------------------===//

#ifndef TC_BASE
#define TC_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// TC Dialect Definition
//===----------------------------------------------------------------------===//

def TC_Dialect : Dialect {
  let name = "tc";
  let cppNamespace = "::mlir::tc";

  let summary = "Tensor Compute Dialect";
  let description = [{
    The Tensor Compute (TC) dialect provides high-level tensor computation
    operations. It focuses on:

    - Tensor contractions (einsum notation)
    - Singular Value Decomposition (SVD) with truncation
    - QR decomposition
    - Tensor manipulations (transpose, reshape, truncate)
    - Support for dynamic tensor dimensions
    - Sparse/dense tensor representations

    The dialect is designed to lower to LinAlg dialect for further optimization
    and code generation.
  }];

  // Use default type parsing/printing
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// TC Operation Base Class
//===----------------------------------------------------------------------===//

// Base class for all TC dialect operations
class TC_Op<string mnemonic, list<Trait> traits = []> :
    Op<TC_Dialect, mnemonic, traits>;

#endif // TC_BASE
