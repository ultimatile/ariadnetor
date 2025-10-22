//===- Passes.h - TC Dialect Transformation Passes -------------*- C++ -*-===//
//
// Tensor Compute Dialect - Transformation Passes
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_COMPUTE_DIALECT_TRANSFORMS_PASSES_H
#define TENSOR_COMPUTE_DIALECT_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tc {

/// Create a pass to convert TC dialect operations to LinAlg dialect
std::unique_ptr<Pass> createConvertTCToLinalgPass();

/// Register all TC dialect transformation passes
void registerTCPasses();

} // namespace tc
} // namespace mlir

#endif // TENSOR_COMPUTE_DIALECT_TRANSFORMS_PASSES_H
