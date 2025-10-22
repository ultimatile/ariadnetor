//===- TCDialect.h - C API for TC Dialect -----------------------*- C -*-===//
//
// Tensor Compute Dialect C API
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_COMPUTE_CAPI_TCDIALECT_H
#define TENSOR_COMPUTE_CAPI_TCDIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for dialect registration
MLIR_CAPI_EXPORTED MlirDialectHandle mlirGetDialectHandle__tc__(void);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_COMPUTE_CAPI_TCDIALECT_H
