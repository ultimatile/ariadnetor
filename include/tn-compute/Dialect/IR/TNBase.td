//===- TNBase.td - Tensor Network Dialect Base ------------*- tablegen -*-===//
//
// Tensor Network Compute Dialect - Base definitions
//
//===----------------------------------------------------------------------===//

#ifndef TN_BASE
#define TN_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// TN Dialect Definition
//===----------------------------------------------------------------------===//

def TN_Dialect : Dialect {
  let name = "tn";
  let cppNamespace = "::mlir::tn";

  let summary = "Tensor Network Compute Dialect";
  let description = [{
    The Tensor Network Compute (TN) dialect provides high-level tensor network
    operations for distributed tensor network computations. It focuses on:

    - Tensor contractions (einsum notation)
    - Singular Value Decomposition (SVD) with truncation
    - QR decomposition
    - Tensor manipulations (transpose, reshape, truncate)
    - Support for dynamic tensor dimensions
    - Sparse/dense tensor representations
    - Future: distributed tensor layouts

    The dialect is designed to lower to LinAlg dialect for further optimization
    and code generation.
  }];

  // Use default type parsing/printing
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// TN Operation Base Class
//===----------------------------------------------------------------------===//

// Base class for all TN dialect operations
class TN_Op<string mnemonic, list<Trait> traits = []> :
    Op<TN_Dialect, mnemonic, traits>;

#endif // TN_BASE
