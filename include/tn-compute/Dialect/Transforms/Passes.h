//===- Passes.h - TN Dialect Transformation Passes -------------*- C++ -*-===//
//
// Tensor Network Compute Dialect - Transformation Passes
//
//===----------------------------------------------------------------------===//

#ifndef TN_COMPUTE_DIALECT_TRANSFORMS_PASSES_H
#define TN_COMPUTE_DIALECT_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tn {

/// Create a pass to convert TN dialect operations to LinAlg dialect
std::unique_ptr<Pass> createConvertTNToLinalgPass();

/// Register all TN dialect transformation passes
void registerTNPasses();

} // namespace tn
} // namespace mlir

#endif // TN_COMPUTE_DIALECT_TRANSFORMS_PASSES_H
