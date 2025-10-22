//===- TCDialect.cpp - C API for TC Dialect --------------------*- C++ -*-===//
//
// Tensor Compute Dialect C API Implementation
//
//===----------------------------------------------------------------------===//

#include "tensor-compute/CAPI/TCDialect.h"
#include "tensor-compute/Dialect/IR/TCDialect.h"

#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TC, tc, mlir::tc::TCDialect)
