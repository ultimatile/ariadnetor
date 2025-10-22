//===- TCDialect.h - Tensor Compute Dialect --------------------*- C++ -*-===//
//
// Tensor Compute Dialect
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_COMPUTE_DIALECT_IR_TCDIALECT_H
#define TENSOR_COMPUTE_DIALECT_IR_TCDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include generated dialect declarations
#include "tensor-compute/Dialect/IR/TCDialect.h.inc"

// Note: No custom type definitions yet - we use MLIR builtin types with constraints
// Future: Include TCTypes.h.inc when TC_TensorType is defined

// Include generated operation declarations
#define GET_OP_CLASSES
#include "tensor-compute/Dialect/IR/TCOps.h.inc"

#endif // TENSOR_COMPUTE_DIALECT_IR_TCDIALECT_H
