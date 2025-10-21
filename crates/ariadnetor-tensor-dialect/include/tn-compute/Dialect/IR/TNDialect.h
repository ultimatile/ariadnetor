//===- TNDialect.h - Tensor Network Dialect --------------------*- C++ -*-===//
//
// Tensor Network Compute Dialect
//
//===----------------------------------------------------------------------===//

#ifndef TN_COMPUTE_DIALECT_IR_TNDIALECT_H
#define TN_COMPUTE_DIALECT_IR_TNDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include generated dialect declarations
#include "tn-compute/Dialect/IR/TNDialect.h.inc"

// Note: No custom type definitions yet - we use MLIR builtin types with constraints
// Future: Include TNTypes.h.inc when TN_TensorType is defined

// Include generated operation declarations
#define GET_OP_CLASSES
#include "tn-compute/Dialect/IR/TNOps.h.inc"

#endif // TN_COMPUTE_DIALECT_IR_TNDIALECT_H
