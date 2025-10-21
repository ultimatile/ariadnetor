//===- TNDialect.h - C API for TN Dialect -----------------------*- C -*-===//
//
// TN-Compute Dialect C API
//
//===----------------------------------------------------------------------===//

#ifndef TN_COMPUTE_CAPI_TNDIALECT_H
#define TN_COMPUTE_CAPI_TNDIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for dialect registration
MLIR_CAPI_EXPORTED MlirDialectHandle mlirGetDialectHandle__tn__(void);

#ifdef __cplusplus
}
#endif

#endif // TN_COMPUTE_CAPI_TNDIALECT_H
