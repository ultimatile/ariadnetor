//===- TCOps.cpp - Tensor Compute Operations -------------------*- C++ -*-===//
//
// Tensor Compute Dialect - Operation implementations
//
//===----------------------------------------------------------------------===//

#include "tensor-compute/Dialect/IR/TCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <regex>

using namespace mlir;
using namespace mlir::tc;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Parse einsum notation: "ij,jk->ik"
static bool parseEinsumNotation(StringRef notation,
                                 SmallVectorImpl<char> &lhsIndices,
                                 SmallVectorImpl<char> &rhsIndices,
                                 SmallVectorImpl<char> &outIndices) {
  std::string str = notation.str();
  std::regex pattern(R"(([a-z]+),([a-z]+)->([a-z]*))");
  std::smatch match;

  if (!std::regex_match(str, match, pattern))
    return false;

  lhsIndices.assign(match[1].str().begin(), match[1].str().end());
  rhsIndices.assign(match[2].str().begin(), match[2].str().end());
  outIndices.assign(match[3].str().begin(), match[3].str().end());

  return true;
}

//===----------------------------------------------------------------------===//
// TC Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensor-compute/Dialect/IR/TCOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ContractOp Verification
//===----------------------------------------------------------------------===//

LogicalResult ContractOp::verify() {
  auto lhsType = mlir::cast<RankedTensorType>(getLhs().getType());
  auto rhsType = mlir::cast<RankedTensorType>(getRhs().getType());
  auto resultType = mlir::cast<RankedTensorType>(getResult().getType());

  StringRef notation = getIndices();
  SmallVector<char> lhsIndices, rhsIndices, outIndices;

  if (!parseEinsumNotation(notation, lhsIndices, rhsIndices, outIndices)) {
    return emitOpError("invalid einsum notation: ") << notation;
  }

  // Verify that the number of indices matches tensor ranks
  if (static_cast<int64_t>(lhsIndices.size()) != lhsType.getRank()) {
    return emitOpError("lhs tensor rank (")
           << lhsType.getRank() << ") doesn't match indices count ("
           << lhsIndices.size() << ")";
  }

  if (static_cast<int64_t>(rhsIndices.size()) != rhsType.getRank()) {
    return emitOpError("rhs tensor rank (")
           << rhsType.getRank() << ") doesn't match indices count ("
           << rhsIndices.size() << ")";
  }

  if (static_cast<int64_t>(outIndices.size()) != resultType.getRank()) {
    return emitOpError("result tensor rank (")
           << resultType.getRank() << ") doesn't match output indices count ("
           << outIndices.size() << ")";
  }

  // TODO: Add more sophisticated verification:
  // - Check dimension compatibility for contracted indices
  // - Verify output dimensions match expected sizes
  // - Validate that all output indices appear in inputs

  return success();
}

//===----------------------------------------------------------------------===//
// SVDOp Verification
//===----------------------------------------------------------------------===//

LogicalResult SVDOp::verify() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());

  // Input must be 2D (matrix)
  if (inputType.getRank() != 2) {
    return emitOpError("input must be a 2D tensor (matrix), got rank ")
           << inputType.getRank();
  }

  // Verify max_chi if present
  if (auto maxChi = getMaxChi()) {
    if (maxChi.value() <= 0) {
      return emitOpError("max_chi must be positive, got ") << maxChi.value();
    }
  }

  // Verify threshold if present
  if (auto threshold = getThreshold()) {
    double thresholdVal = threshold.value().convertToDouble();
    if (thresholdVal < 0.0) {
      return emitOpError("threshold must be non-negative, got ")
             << thresholdVal;
    }
  }

  auto uType = mlir::cast<RankedTensorType>(getU().getType());
  auto sType = mlir::cast<RankedTensorType>(getS().getType());
  auto vType = mlir::cast<RankedTensorType>(getV().getType());

  // U and V must be 2D
  if (uType.getRank() != 2) {
    return emitOpError("U must be a 2D tensor, got rank ") << uType.getRank();
  }

  if (vType.getRank() != 2) {
    return emitOpError("V must be a 2D tensor, got rank ") << vType.getRank();
  }

  // S must be 1D (vector of singular values)
  if (sType.getRank() != 1) {
    return emitOpError("S must be a 1D tensor (vector), got rank ")
           << sType.getRank();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// QROp Verification
//===----------------------------------------------------------------------===//

LogicalResult QROp::verify() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());

  // Input must be 2D (matrix)
  if (inputType.getRank() != 2) {
    return emitOpError("input must be a 2D tensor (matrix), got rank ")
           << inputType.getRank();
  }

  auto qType = mlir::cast<RankedTensorType>(getQ().getType());
  auto rType = mlir::cast<RankedTensorType>(getR().getType());

  // Q and R must be 2D
  if (qType.getRank() != 2) {
    return emitOpError("Q must be a 2D tensor, got rank ") << qType.getRank();
  }

  if (rType.getRank() != 2) {
    return emitOpError("R must be a 2D tensor, got rank ") << rType.getRank();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp Verification
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  auto resultType = mlir::cast<RankedTensorType>(getResult().getType());

  auto permutation = getPermutation();
  int64_t rank = inputType.getRank();

  // Verify permutation size matches rank
  if (static_cast<int64_t>(permutation.size()) != rank) {
    return emitOpError("permutation size (")
           << permutation.size() << ") doesn't match tensor rank (" << rank
           << ")";
  }

  // Verify permutation is a valid permutation [0, rank)
  SmallVector<bool> seen(rank, false);
  for (auto idx : permutation) {
    int64_t i = mlir::cast<IntegerAttr>(idx).getInt();
    if (i < 0 || i >= rank) {
      return emitOpError("permutation index ") << i << " out of range [0, "
                                                << rank << ")";
    }
    if (seen[i]) {
      return emitOpError("duplicate index ") << i << " in permutation";
    }
    seen[i] = true;
  }

  // Verify result type matches permuted input type
  if (resultType.getRank() != rank) {
    return emitOpError("result rank doesn't match input rank");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp Verification
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  auto resultType = mlir::cast<RankedTensorType>(getResult().getType());

  // Calculate total elements in input (if static)
  int64_t inputElements = 1;
  bool inputStatic = true;
  for (int64_t dim : inputType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      inputStatic = false;
      break;
    }
    inputElements *= dim;
  }

  // Calculate total elements in output (if static)
  int64_t outputElements = 1;
  bool outputStatic = true;
  int dynamicCount = 0;
  for (int64_t dim : resultType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      outputStatic = false;
      dynamicCount++;
    } else {
      outputElements *= dim;
    }
  }

  // At most one dynamic dimension in output
  if (dynamicCount > 1) {
    return emitOpError(
        "reshape can have at most one dynamic dimension in output");
  }

  // If both static, verify element count matches
  if (inputStatic && outputStatic && inputElements != outputElements) {
    return emitOpError("total element count must be preserved: input has ")
           << inputElements << " elements, output has " << outputElements;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TruncateOp Verification
//===----------------------------------------------------------------------===//

LogicalResult TruncateOp::verify() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());

  // Input must be 2D for truncation (typically operates on matrices)
  if (inputType.getRank() != 2) {
    return emitOpError("input must be a 2D tensor (matrix), got rank ")
           << inputType.getRank();
  }

  // At least one of max_chi or threshold must be specified
  if (!getMaxChi() && !getThreshold()) {
    return emitOpError("at least one of max_chi or threshold must be specified");
  }

  // Verify max_chi if present
  if (auto maxChi = getMaxChi()) {
    if (maxChi.value() <= 0) {
      return emitOpError("max_chi must be positive, got ") << maxChi.value();
    }
  }

  // Verify threshold if present
  if (auto threshold = getThreshold()) {
    double thresholdVal = threshold.value().convertToDouble();
    if (thresholdVal < 0.0) {
      return emitOpError("threshold must be non-negative, got ")
             << thresholdVal;
    }
  }

  return success();
}
