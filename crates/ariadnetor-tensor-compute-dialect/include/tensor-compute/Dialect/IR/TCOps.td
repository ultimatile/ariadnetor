//===- TCOps.td - Tensor Compute Dialect Operations -------*- tablegen -*-===//
//
// Tensor Compute Dialect - Operation definitions
//
//===----------------------------------------------------------------------===//

#ifndef TC_OPS
#define TC_OPS

include "tensor-compute/Dialect/IR/TCBase.td"
include "tensor-compute/Dialect/IR/TCTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Tensor Contraction Operations
//===----------------------------------------------------------------------===//

def TC_ContractOp : TC_Op<"contract", [Pure]> {
  let summary = "Tensor contraction operation (einsum)";

  let description = [{
    Performs tensor contraction following einsum notation.

    Examples:
      - Matrix multiplication: "ij,jk->ik"
      - Tensor dot product: "ijk,ijk->"
      - Outer product: "i,j->ij"
      - Trace: "ii->"
      - High-dimensional: "ijkl,klmn->ijmn"

    The operation supports dynamic tensor shapes and will be lowered
    to appropriate LinAlg operations (matmul, generic) during compilation.

    Example:
    ```mlir
    %C = tc.contract %A, %B {
        indices = "ij,jk->ik"
    } : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
    ```
  }];

  let arguments = (ins
    TC_Tensor:$lhs,
    TC_Tensor:$rhs,
    StrAttr:$indices
  );

  let results = (outs TC_Tensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs
    `{` `indices` `=` $indices `}`
    attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Matrix Decomposition Operations
//===----------------------------------------------------------------------===//

def TC_SVDOp : TC_Op<"svd", []> {
  let summary = "Singular Value Decomposition";

  let description = [{
    Performs Singular Value Decomposition (SVD) with optional truncation.

    Given a matrix A (m x n), computes:
      A = U * S * V^T
    where:
      - U: left singular vectors (m x k)
      - S: singular values (k)
      - V: right singular vectors (n x k)
      - k = min(m, n) without truncation

    Optional truncation parameters:
      - max_chi: maximum bond dimension (caps k)
      - threshold: discard singular values below this threshold

    This is a critical operation for tensor networks, used in:
      - Matrix Product States (MPS)
      - Tensor decomposition
      - Bond dimension reduction

    Example:
    ```mlir
    %U, %S, %V = tc.svd %A {
        max_chi = 100,
        threshold = 1.0e-10
    } : tensor<?x?xf64> -> (tensor<?x?xf64>, tensor<?xf64>, tensor<?x?xf64>)
    ```
  }];

  let arguments = (ins
    TC_FloatTensor:$input,
    OptionalAttr<I64Attr>:$max_chi,
    OptionalAttr<F64Attr>:$threshold
  );

  let results = (outs
    TC_FloatTensor:$U,
    TC_Vector:$S,
    TC_FloatTensor:$V
  );

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` `(` type($U) `,` type($S) `,` type($V) `)`
  }];

  let hasVerifier = 1;
}

def TC_QROp : TC_Op<"qr", []> {
  let summary = "QR decomposition";

  let description = [{
    Performs QR decomposition on a matrix.

    Given a matrix A (m x n), computes:
      A = Q * R
    where:
      - Q: orthogonal matrix (m x min(m,n))
      - R: upper triangular matrix (min(m,n) x n)

    QR decomposition is used in tensor networks for:
      - Orthogonalization of MPS tensors
      - Canonical forms
      - Numerical stability

    Example:
    ```mlir
    %Q, %R = tc.qr %A : tensor<?x?xf64> -> (tensor<?x?xf64>, tensor<?x?xf64>)
    ```
  }];

  let arguments = (ins TC_FloatTensor:$input);

  let results = (outs
    TC_FloatTensor:$Q,
    TC_FloatTensor:$R
  );

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` `(` type($Q) `,` type($R) `)`
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tensor Manipulation Operations
//===----------------------------------------------------------------------===//

def TC_TransposeOp : TC_Op<"transpose", [Pure]> {
  let summary = "Tensor transpose operation";

  let description = [{
    Transposes a tensor according to the specified permutation.

    The permutation array specifies how to reorder the dimensions.
    For example, [1, 0] swaps the first two dimensions of a 2D tensor.

    Example:
    ```mlir
    // Transpose a matrix
    %transposed = tc.transpose %A {
        permutation = [1, 0]
    } : tensor<?x?xf64> -> tensor<?x?xf64>

    // Permute a 4D tensor
    %permuted = tc.transpose %T {
        permutation = [0, 2, 1, 3]
    } : tensor<?x?x?x?xf64> -> tensor<?x?x?x?xf64>
    ```
  }];

  let arguments = (ins
    TC_Tensor:$input,
    I64ArrayAttr:$permutation
  );

  let results = (outs TC_Tensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
}

def TC_ReshapeOp : TC_Op<"reshape", [Pure]> {
  let summary = "Tensor reshape operation";

  let description = [{
    Reshapes a tensor to a new shape while preserving the total number
    of elements.

    The new shape can contain at most one dynamic dimension (-1 or ?).

    Example:
    ```mlir
    // Flatten to vector
    %flat = tc.reshape %A {
        new_shape = [1000]
    } : tensor<10x100xf64> -> tensor<1000xf64>

    // Reshape to matrix
    %mat = tc.reshape %v {
        new_shape = [100, 10]
    } : tensor<1000xf64> -> tensor<100x10xf64>
    ```
  }];

  let arguments = (ins
    TC_Tensor:$input,
    I64ArrayAttr:$new_shape
  );

  let results = (outs TC_Tensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
}

def TC_TruncateOp : TC_Op<"truncate", []> {
  let summary = "Bond dimension truncation";

  let description = [{
    Truncates bond dimensions of a tensor based on singular values.

    This operation is typically used after SVD to reduce the bond dimension
    while preserving the most important information. It can:
      - Limit the maximum bond dimension (max_chi)
      - Discard small singular values (threshold)
      - Or both

    The actual truncation depends on the singular value spectrum of the
    input tensor. This operation may internally perform SVD if needed.

    Example:
    ```mlir
    %truncated = tc.truncate %tensor {
        max_chi = 50,
        threshold = 1.0e-8
    } : tensor<?x?xf64> -> tensor<?x?xf64>
    ```
  }];

  let arguments = (ins
    TC_FloatTensor:$input,
    OptionalAttr<I64Attr>:$max_chi,
    OptionalAttr<F64Attr>:$threshold
  );

  let results = (outs TC_FloatTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Batch Operations (Future)
//===----------------------------------------------------------------------===//

// TODO: Implement batch operations for efficient parallel execution
//
// def TC_BatchContractOp : TC_Op<"batch_contract", [Pure]> {
//   let summary = "Batch tensor contraction";
//   let description = [{
//     Performs multiple tensor contractions in parallel.
//     Useful for processing multiple tensor pairs efficiently.
//   }];
//   // ...
// }

#endif // TC_OPS
