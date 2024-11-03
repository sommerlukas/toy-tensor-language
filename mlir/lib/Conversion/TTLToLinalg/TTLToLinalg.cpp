#include "Conversion/TTLToTensor/TTLToTensor.h"

#include "../TTLTypeConverter.h"
#include "Dialect/TTL/TTLDialect.h"
#include "Dialect/TTL/TTLOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace ttl {
#define GEN_PASS_DEF_TTLTOLINALG
#include "Conversion/Passes.h.inc"
} // namespace ttl
} // namespace mlir

using namespace mlir;
using namespace mlir::ttl;

#define DEBUG_TYPE "ttl-to-linalg"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace {

Value getSizeInDim(Location Loc, Value tensor, int64_t dim,
                   ConversionPatternRewriter &rewriter) {
  Value idx =
      rewriter.create<index::ConstantOp>(Loc, rewriter.getIndexAttr(dim));
  Value dynSize = rewriter.create<tensor::DimOp>(Loc, tensor, idx);
  return dynSize;
}

template <typename ToReplaceOp, typename ReplacementOp>
struct TensorBinaryOpLowering : public OpConversionPattern<ToReplaceOp> {

  using OpConversionPattern<ToReplaceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ToReplaceOp op, typename ToReplaceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto Loc = op.getLoc();
    auto resultTy = dyn_cast<RankedTensorType>(
        this->typeConverter->convertType(op.getResult().getType()));
    if (!resultTy) {
      // Pattern doesn't apply if the result isn't tensor. Scalar (int, float)
      // arithmetic is handled by a different pass.
      return failure();
    }
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < resultTy.getRank(); ++i) {
      if (resultTy.isDynamicDim(i)) {
        // If this dimension is static, we just query the left operand for it's
        // dynamic size in that dimension.
        dynamicSizes.push_back(
            getSizeInDim(Loc, adaptor.getLeft(), i, rewriter));
      }
    }
    Value emptyTensor =
        rewriter.create<tensor::EmptyOp>(Loc, resultTy, dynamicSizes);
    rewriter.replaceOpWithNewOp<ReplacementOp>(
        op, resultTy, adaptor.getOperands(), emptyTensor);
    return success();
  }
};

struct MatmulLowering : public OpConversionPattern<ttl::MatMul> {

  using OpConversionPattern<ttl::MatMul>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MatMul op, MatMul::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto Loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(
        typeConverter->convertType(op.getResult().getType()));
    assert(resultTy.getRank() == 2 &&
           "Only two-dimensional matrix multiplication supported");
    SmallVector<Value> dynamicSizes;
    // We multiply a N x K tensor with a K x M tensor, resulting in N x M.
    // In the dynamic case, N can be queried as the first dimension of the left
    // operand and M can be queried as the second dimension of the right
    // operand.
    if (resultTy.isDynamicDim(0)) {
      dynamicSizes.push_back(getSizeInDim(Loc, adaptor.getLeft(), 0, rewriter));
    }
    if (resultTy.isDynamicDim(1)) {
      dynamicSizes.push_back(
          getSizeInDim(Loc, adaptor.getRight(), 1, rewriter));
    }
    Value emptyTensor =
        rewriter.create<tensor::EmptyOp>(Loc, resultTy, dynamicSizes);
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        op, resultTy, adaptor.getOperands(), emptyTensor);
    return success();
  }
};

struct ConvertTTLToLinalgPass
    : public ttl::impl::TTLToLinalgBase<ConvertTTLToLinalgPass> {

  using ttl::impl::TTLToLinalgBase<ConvertTTLToLinalgPass>::TTLToLinalgBase;

  void runOnOperation() override;
};

} // namespace

void ConvertTTLToLinalgPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<tensor::TensorDialect, scf::SCFDialect, BuiltinDialect,
                         func::FuncDialect, index::IndexDialect,
                         arith::ArithDialect, linalg::LinalgDialect>();

  target.addLegalDialect<ttl::TTLDialect>();
  target.addDynamicallyLegalOp<ttl::Add, ttl::Sub, ttl::Mul, ttl::Div>(
      [](Operation *op) {
        return !isa<ttl::TensorType>(op->getResults().front().getType());
      });
  target.addIllegalOp<ttl::MatMul>();

  TTLTypeConverter typeConverter;

  RewritePatternSet patterns(&getContext());
  patterns.add<TensorBinaryOpLowering<ttl::Add, linalg::AddOp>,
               TensorBinaryOpLowering<ttl::Sub, linalg::SubOp>,
               TensorBinaryOpLowering<ttl::Mul, linalg::MulOp>,
               TensorBinaryOpLowering<ttl::Div, linalg::DivOp>, MatmulLowering>(
      typeConverter, &getContext());

  ModuleOp Module = getOperation();

  if (failed(applyPartialConversion(Module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}
