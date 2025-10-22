//===- TCDialect.cpp - Tensor Compute Dialect ------------------*- C++ -*-===//
//
// Tensor Compute Dialect - Implementation
//
//===----------------------------------------------------------------------===//

#include "tensor-compute/Dialect/IR/TCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::tc;

//===----------------------------------------------------------------------===//
// TC Dialect
//===----------------------------------------------------------------------===//

#include "tensor-compute/Dialect/IR/TCDialect.cpp.inc"

void TCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tensor-compute/Dialect/IR/TCOps.cpp.inc"
      >();

  // Future: register types when we implement custom TensorType
  // addTypes<
  // #define GET_TYPEDEF_LIST
  // #include "tensor-compute/Dialect/IR/TCTypes.cpp.inc"
  //     >();
}

//===----------------------------------------------------------------------===//
// Attribute/Type Parsing and Printing
//===----------------------------------------------------------------------===//

// Parse an attribute
// Since we don't have custom attributes yet, delegate to default parser
Attribute TCDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  return Attribute();  // Return null - no custom attributes defined
}

// Print an attribute
// Since we don't have custom attributes yet, this should not be called
void TCDialect::printAttribute(Attribute attr,
                                DialectAsmPrinter &printer) const {
  llvm_unreachable("no custom attributes in TC dialect");
}

// Parse a type
// Since we don't have custom types yet, delegate to default parser
Type TCDialect::parseType(DialectAsmParser &parser) const {
  return Type();  // Return null - no custom types defined
}

// Print a type
// Since we don't have custom types yet, this should not be called
void TCDialect::printType(Type type, DialectAsmPrinter &printer) const {
  llvm_unreachable("no custom types in TC dialect");
}
