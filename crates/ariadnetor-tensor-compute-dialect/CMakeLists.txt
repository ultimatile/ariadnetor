cmake_minimum_required(VERSION 3.20)
project(tensor-compute-dialect LANGUAGES CXX C)

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

# Create output directory for TableGen
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/tensor-compute/Dialect/IR)

# TableGen for TC Dialect
set(LLVM_TARGET_DEFINITIONS include/tensor-compute/Dialect/IR/TCOps.td)
mlir_tablegen(include/tensor-compute/Dialect/IR/TCOps.h.inc -gen-op-decls)
mlir_tablegen(include/tensor-compute/Dialect/IR/TCOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/tensor-compute/Dialect/IR/TCDialect.h.inc -gen-dialect-decls -dialect=tc)
mlir_tablegen(include/tensor-compute/Dialect/IR/TCDialect.cpp.inc -gen-dialect-defs -dialect=tc)
# Note: No typedef generation needed - we use MLIR builtin types with constraints
# Future: Add when TC_TensorType is defined with TypeDef
add_public_tablegen_target(TCOpsIncGen)

# Add TCDialect library
# Using static library (CAPI auto-registration macro removed to avoid init conflicts)
add_mlir_dialect_library(MLIRTCDialect
  lib/Dialect/IR/TCDialect.cpp
  lib/Dialect/IR/TCOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tensor-compute

  DEPENDS
  TCOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
)

# Add TCTransforms library
add_mlir_dialect_library(MLIRTCTransforms
  lib/Dialect/Transforms/ConvertTCToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tensor-compute

  LINK_LIBS PUBLIC
  MLIRTCDialect
  MLIRLinalgDialect
  MLIRTransforms
  MLIRPass
)

# Add TCCAPI library
add_mlir_dialect_library(MLIRTCCAPI
  lib/CAPI/TCDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tensor-compute

  LINK_LIBS PUBLIC
  MLIRTCDialect
  MLIRCAPIIR
)

# Installation
install(TARGETS MLIRTCDialect MLIRTCTransforms MLIRTCCAPI
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.td")

install(DIRECTORY ${PROJECT_BINARY_DIR}/include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h.inc")
