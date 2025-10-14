//===- ConvertTNToLinalg.cpp - Convert TN to LinAlg ------------*- C++ -*-===//
//
// Tensor Network Compute Dialect - Conversion to LinAlg
//
//===----------------------------------------------------------------------===//

#include "tn-compute/Dialect/IR/TNDialect.h"
#include "tn-compute/Dialect/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tn;

//===----------------------------------------------------------------------===//
// Pattern: TN Contract to LinAlg MatMul
//===----------------------------------------------------------------------===//

namespace {

struct ContractOpToMatMulPattern : public OpRewritePattern<ContractOp> {
  using OpRewritePattern<ContractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ContractOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if this is a simple matrix multiplication "ij,jk->ik"
    StringRef indices = op.getIndices();

    if (indices == "ij,jk->ik") {
      // This is matrix multiplication, convert to linalg.matmul
      auto lhs = op.getLhs();
      auto rhs = op.getRhs();
      auto resultType = op.getResult().getType();

      // Create empty output tensor
      auto loc = op.getLoc();
      auto emptyOp = rewriter.create<tensor::EmptyOp>(
          loc, resultType.cast<RankedTensorType>().getShape(),
          resultType.cast<RankedTensorType>().getElementType());

      // Create linalg.matmul
      auto matmulOp = rewriter.create<linalg::MatmulOp>(
          loc, ValueRange{lhs, rhs}, ValueRange{emptyOp.getResult()});

      rewriter.replaceOp(op, matmulOp.getResult(0));
      return success();
    }

    // TODO: Handle other einsum patterns
    // - Batched matmul: "bij,bjk->bik"
    // - Tensor dot: "ijk,ijk->"
    // - Outer product: "i,j->ij"
    // - General contractions via linalg.generic

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: TN Transpose to LinAlg Transpose
//===----------------------------------------------------------------------===//

struct TransposeOpToLinalgPattern : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                 PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inputType = op.getInput().getType().cast<RankedTensorType>();
    auto resultType = op.getResult().getType().cast<RankedTensorType>();

    // Create empty output tensor
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    // Create linalg.transpose
    auto permutation = op.getPermutation();
    SmallVector<int64_t> permVec;
    for (auto attr : permutation) {
      permVec.push_back(attr.cast<IntegerAttr>().getInt());
    }

    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, op.getInput(), emptyOp.getResult(), permVec);

    rewriter.replaceOp(op, transposeOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: TN SVD to Runtime Function Call
//===----------------------------------------------------------------------===//

struct SVDOpToRuntimeCallPattern : public OpRewritePattern<SVDOp> {
  using OpRewritePattern<SVDOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SVDOp op,
                                 PatternRewriter &rewriter) const override {
    // SVD requires calling external linear algebra libraries (SLATE, LAPACK)
    // We lower this to a runtime function call that will be linked later

    auto loc = op.getLoc();
    auto context = rewriter.getContext();

    // Declare runtime function signature:
    // func @tn_runtime_svd(tensor<?x?xf64>, i64, f64)
    //     -> (tensor<?x?xf64>, tensor<?xf64>, tensor<?x?xf64>)

    auto module = op->getParentOfType<ModuleOp>();
    auto symbolTable = SymbolTable(module);

    // Check if function already declared
    FlatSymbolRefAttr funcRef;
    if (auto func = symbolTable.lookup<func::FuncOp>("tn_runtime_svd")) {
      funcRef = SymbolRefAttr::get(context, "tn_runtime_svd");
    } else {
      // Declare the function
      auto f64Type = Float64Type::get(context);
      auto i64Type = IntegerType::get(context, 64);

      auto inputType = op.getInput().getType();
      auto uType = op.getU().getType();
      auto sType = op.getS().getType();
      auto vType = op.getV().getType();

      auto funcType = FunctionType::get(
          context,
          {inputType, i64Type, f64Type}, // input, max_chi, threshold
          {uType, sType, vType}          // U, S, V
      );

      auto funcOp = func::FuncOp::create(loc, "tn_runtime_svd", funcType);
      funcOp.setPrivate();
      symbolTable.insert(funcOp);
      funcRef = SymbolRefAttr::get(context, "tn_runtime_svd");
    }

    // Create function call
    SmallVector<Value> operands;
    operands.push_back(op.getInput());

    // Add max_chi parameter (0 means no limit)
    auto maxChi = op.getMaxChi().value_or(0);
    operands.push_back(rewriter.create<arith::ConstantIntOp>(loc, maxChi, 64));

    // Add threshold parameter (0.0 means no threshold)
    auto threshold = op.getThreshold().value_or(0.0);
    operands.push_back(
        rewriter.create<arith::ConstantFloatOp>(loc, APFloat(threshold),
                                                 Float64Type::get(context)));

    auto callOp =
        rewriter.create<func::CallOp>(loc, funcRef, op.getResultTypes(), operands);

    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: TN QR to Runtime Function Call
//===----------------------------------------------------------------------===//

struct QROpToRuntimeCallPattern : public OpRewritePattern<QROp> {
  using OpRewritePattern<QROp>::OpRewritePattern;

  LogicalResult matchAndRewrite(QROp op,
                                 PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto context = rewriter.getContext();

    auto module = op->getParentOfType<ModuleOp>();
    auto symbolTable = SymbolTable(module);

    // Declare runtime function: func @tn_runtime_qr(tensor<?x?xf64>)
    //                                -> (tensor<?x?xf64>, tensor<?x?xf64>)
    FlatSymbolRefAttr funcRef;
    if (auto func = symbolTable.lookup<func::FuncOp>("tn_runtime_qr")) {
      funcRef = SymbolRefAttr::get(context, "tn_runtime_qr");
    } else {
      auto inputType = op.getInput().getType();
      auto qType = op.getQ().getType();
      auto rType = op.getR().getType();

      auto funcType =
          FunctionType::get(context, {inputType}, {qType, rType});

      auto funcOp = func::FuncOp::create(loc, "tn_runtime_qr", funcType);
      funcOp.setPrivate();
      symbolTable.insert(funcOp);
      funcRef = SymbolRefAttr::get(context, "tn_runtime_qr");
    }

    auto callOp = rewriter.create<func::CallOp>(loc, funcRef,
                                                  op.getResultTypes(),
                                                  op.getInput());

    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertTNToLinalg Pass
//===----------------------------------------------------------------------===//

struct ConvertTNToLinalgPass
    : public PassWrapper<ConvertTNToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTNToLinalgPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect, func::FuncDialect>();
  }

  StringRef getArgument() const final { return "convert-tn-to-linalg"; }

  StringRef getDescription() const final {
    return "Convert TN dialect operations to LinAlg dialect";
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<ContractOpToMatMulPattern, TransposeOpToLinalgPattern,
                 SVDOpToRuntimeCallPattern, QROpToRuntimeCallPattern>(context);

    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                           arith::ArithDialect, func::FuncDialect>();
    target.addIllegalDialect<TNDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::tn::createConvertTNToLinalgPass() {
  return std::make_unique<ConvertTNToLinalgPass>();
}

void mlir::tn::registerTNPasses() {
  PassRegistration<ConvertTNToLinalgPass>();
}
