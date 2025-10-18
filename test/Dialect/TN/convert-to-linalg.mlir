// RUN: tn-opt %s -convert-tn-to-linalg | FileCheck %s

// Test: Matrix multiplication lowering
// CHECK-LABEL: func.func @test_matmul
func.func @test_matmul(%arg0: tensor<10x20xf64>, %arg1: tensor<20x30xf64>) -> tensor<10x30xf64> {
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<10x30xf64>
  // CHECK: %[[RESULT:.*]] = linalg.matmul ins(%arg0, %arg1 : tensor<10x20xf64>, tensor<20x30xf64>) outs(%[[EMPTY]] : tensor<10x30xf64>) -> tensor<10x30xf64>
  // CHECK: return %[[RESULT]]
  %0 = "tn.contract"(%arg0, %arg1) {indices = "ij,jk->ik"} : (tensor<10x20xf64>, tensor<20x30xf64>) -> tensor<10x30xf64>
  return %0 : tensor<10x30xf64>
}

// Test: Batched matrix multiplication lowering
// CHECK-LABEL: func.func @test_batch_matmul
func.func @test_batch_matmul(%arg0: tensor<32x10x20xf64>, %arg1: tensor<32x20x30xf64>) -> tensor<32x10x30xf64> {
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x10x30xf64>
  // CHECK: %[[RESULT:.*]] = linalg.batch_matmul ins(%arg0, %arg1 : tensor<32x10x20xf64>, tensor<32x20x30xf64>) outs(%[[EMPTY]] : tensor<32x10x30xf64>) -> tensor<32x10x30xf64>
  // CHECK: return %[[RESULT]]
  %0 = "tn.contract"(%arg0, %arg1) {indices = "bij,bjk->bik"} : (tensor<32x10x20xf64>, tensor<32x20x30xf64>) -> tensor<32x10x30xf64>
  return %0 : tensor<32x10x30xf64>
}

// Test: Element-wise multiplication lowering
// CHECK-LABEL: func.func @test_element_wise
func.func @test_element_wise(%arg0: tensor<10x20xf64>, %arg1: tensor<10x20xf64>) -> tensor<10x20xf64> {
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<10x20xf64>
  // CHECK: %[[RESULT:.*]] = linalg.map { arith.mulfop } ins(%arg0, %arg1 : tensor<10x20xf64>, tensor<10x20xf64>) outs(%[[EMPTY]] : tensor<10x20xf64>)
  // CHECK: return %[[RESULT]]
  %0 = "tn.contract"(%arg0, %arg1) {indices = "ij,ij->ij"} : (tensor<10x20xf64>, tensor<10x20xf64>) -> tensor<10x20xf64>
  return %0 : tensor<10x20xf64>
}

// Test: Transpose lowering
// CHECK-LABEL: func.func @test_transpose
func.func @test_transpose(%arg0: tensor<10x20xf64>) -> tensor<20x10xf64> {
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<20x10xf64>
  // CHECK: %[[RESULT:.*]] = linalg.transpose ins(%arg0 : tensor<10x20xf64>) outs(%[[EMPTY]] : tensor<20x10xf64>) permutation = [1, 0]
  // CHECK: return %[[RESULT]]
  %0 = "tn.transpose"(%arg0) {permutation = array<i64: 1, 0>} : (tensor<10x20xf64>) -> tensor<20x10xf64>
  return %0 : tensor<20x10xf64>
}

// Test: Mixed operations
// CHECK-LABEL: func.func @test_mixed_ops
func.func @test_mixed_ops(%arg0: tensor<5x10xf64>, %arg1: tensor<10x15xf64>) -> tensor<15x5xf64> {
  // First: matrix multiply
  // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<5x15xf64>
  // CHECK: %[[MATMUL:.*]] = linalg.matmul
  %0 = "tn.contract"(%arg0, %arg1) {indices = "ij,jk->ik"} : (tensor<5x10xf64>, tensor<10x15xf64>) -> tensor<5x15xf64>

  // Then: transpose
  // CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<15x5xf64>
  // CHECK: %[[TRANSPOSE:.*]] = linalg.transpose ins(%[[MATMUL]] : tensor<5x15xf64>) outs(%[[EMPTY2]] : tensor<15x5xf64>) permutation = [1, 0]
  %1 = "tn.transpose"(%0) {permutation = array<i64: 1, 0>} : (tensor<5x15xf64>) -> tensor<15x5xf64>

  // CHECK: return %[[TRANSPOSE]]
  return %1 : tensor<15x5xf64>
}
