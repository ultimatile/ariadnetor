//===- TNDialect.cpp - Tensor Network Dialect ------------------*- C++ -*-===//
//
// Tensor Network Compute Dialect - Implementation
//
//===----------------------------------------------------------------------===//

#include "tn-compute/Dialect/IR/TNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::tn;

//===----------------------------------------------------------------------===//
// TN Dialect
//===----------------------------------------------------------------------===//

#include "tn-compute/Dialect/IR/TNDialect.cpp.inc"

void TNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tn-compute/Dialect/IR/TNOps.cpp.inc"
      >();

  // Future: register types when we implement custom TensorType
  // addTypes<
  // #define GET_TYPEDEF_LIST
  // #include "tn-compute/Dialect/IR/TNTypes.cpp.inc"
  //     >();
}
