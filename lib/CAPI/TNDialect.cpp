//===- TNDialect.cpp - C API for TN Dialect --------------------*- C++ -*-===//
//
// TN-Compute Dialect C API Implementation
//
//===----------------------------------------------------------------------===//

#include "tn-compute/CAPI/TNDialect.h"
#include "tn-compute/Dialect/IR/TNDialect.h"

#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TN, tn, mlir::tn::TNDialect)
