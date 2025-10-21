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

//===----------------------------------------------------------------------===//
// Attribute/Type Parsing and Printing
//===----------------------------------------------------------------------===//

// Parse an attribute
// Since we don't have custom attributes yet, delegate to default parser
Attribute TNDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  return Attribute();  // Return null - no custom attributes defined
}

// Print an attribute
// Since we don't have custom attributes yet, this should not be called
void TNDialect::printAttribute(Attribute attr,
                                DialectAsmPrinter &printer) const {
  llvm_unreachable("no custom attributes in TN dialect");
}

// Parse a type
// Since we don't have custom types yet, delegate to default parser
Type TNDialect::parseType(DialectAsmParser &parser) const {
  return Type();  // Return null - no custom types defined
}

// Print a type
// Since we don't have custom types yet, this should not be called
void TNDialect::printType(Type type, DialectAsmPrinter &printer) const {
  llvm_unreachable("no custom types in TN dialect");
}
