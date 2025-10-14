cmake_minimum_required(VERSION 3.20)
project(tn-compute-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Find MLIR
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# TableGen for TN Dialect
set(LLVM_TARGET_DEFINITIONS include/tn-compute/Dialect/IR/TNOps.td)
mlir_tablegen(include/tn-compute/Dialect/IR/TNOps.h.inc -gen-op-decls)
mlir_tablegen(include/tn-compute/Dialect/IR/TNOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/tn-compute/Dialect/IR/TNDialect.h.inc -gen-dialect-decls -dialect=tn)
mlir_tablegen(include/tn-compute/Dialect/IR/TNDialect.cpp.inc -gen-dialect-defs -dialect=tn)
# Note: No typedef generation needed - we use MLIR builtin types with constraints
# Future: Add when TN_TensorType is defined with TypeDef
add_public_tablegen_target(TNOpsIncGen)

# Add TN Dialect library
add_mlir_dialect_library(MLIRTNDialect
  lib/Dialect/IR/TNDialect.cpp
  lib/Dialect/IR/TNOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tn-compute

  DEPENDS
  TNOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
)

# Add TN Transforms library
add_mlir_dialect_library(MLIRTNTransforms
  lib/Dialect/Transforms/ConvertTNToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tn-compute

  LINK_LIBS PUBLIC
  MLIRTNDialect
  MLIRLinalgDialect
  MLIRTransforms
  MLIRPass
)

# Installation
install(TARGETS MLIRTNDialect MLIRTNTransforms
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.td")

install(DIRECTORY ${PROJECT_BINARY_DIR}/include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h.inc")
