//===- TNDialect.h - Tensor Network Dialect --------------------*- C++ -*-===//
//
// Tensor Network Compute Dialect
//
//===----------------------------------------------------------------------===//

#ifndef TN_COMPUTE_DIALECT_IR_TNDIALECT_H
#define TN_COMPUTE_DIALECT_IR_TNDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include generated dialect declarations
#include "tn-compute/Dialect/IR/TNDialect.h.inc"

// Include generated type declarations
#define GET_TYPEDEF_CLASSES
#include "tn-compute/Dialect/IR/TNTypes.h.inc"

// Include generated operation declarations
#define GET_OP_CLASSES
#include "tn-compute/Dialect/IR/TNOps.h.inc"

#endif // TN_COMPUTE_DIALECT_IR_TNDIALECT_H
