#include "Dialect/TTL/TTLOps.h"
#include "Dialect/TTL/TTLAttributes.h"
#include "Dialect/TTL/TTLDialect.h"
#include "Dialect/TTL/TTLTypes.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Dialect/TTL/TTLOps.cpp.inc"

using namespace mlir;
using namespace mlir::ttl;

void ForLoop::build(OpBuilder &builder, OperationState &result,
                    Value lowerBound, Value upperBound, Value step,
                    ValueRange initArgs) {
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands({lowerBound, upperBound, step});
  result.addOperands(initArgs);

  Type lowerBoundTy = lowerBound.getType();

  // Initialize the for-loop with a region and the single block for the loop
  // body.
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  bodyBlock->addArgument(lowerBoundTy, result.location);
  for (Value v : initArgs) {
    result.addTypes(v.getType());
    bodyBlock->addArgument(v.getType(), v.getLoc());
  }
}

OpFoldResult IntConstant::fold(FoldAdaptor adaptor) {
  return adaptor.getConstValAttr();
}

OpFoldResult FloatConstant::fold(FoldAdaptor adaptor) {
  return adaptor.getConstValAttr();
}

namespace {

struct SliceCanonicalizer : public OpRewritePattern<ttl::Slice> {
  using OpRewritePattern<ttl::Slice>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttl::Slice op,
                                PatternRewriter &rewriter) const override {
    if (!op.getTensor().getDefiningOp()) {
      return rewriter.notifyMatchFailure(op, "Unknown matrix");
    }
    auto scalarFill =
        dyn_cast<ttl::TensorScalarInit>(op.getTensor().getDefiningOp());
    if (!scalarFill) {
      return rewriter.notifyMatchFailure(op, "Not a scalar initializer");
    }
    for (auto S : op.getSizes()) {
      if (!S.getDefiningOp()) {
        return rewriter.notifyMatchFailure(op, "Unknown size");
      }
      auto constInt = dyn_cast<ttl::IntConstant>(S.getDefiningOp());
      if (!constInt) {
        return rewriter.notifyMatchFailure(op, "Not a constant size");
      }
      if (constInt.getConstVal() != 1) {
        return rewriter.notifyMatchFailure(op, "Not size 1");
      }
    }
    rewriter.replaceOp(op, scalarFill.getInitVal());
    return success();
  }
};
} // namespace

void Slice::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add(std::make_unique<SliceCanonicalizer>(context));
}

OpFoldResult Dim::fold(FoldAdaptor adaptor) {
  if (!adaptor.getDimension()) {
    return nullptr;
  }
  IntegerAttr constDim = dyn_cast<IntegerAttr>(adaptor.getDimension());
  if (!constDim) {
    return nullptr;
  }
  auto tensorTy = cast<ttl::TensorType>(getTensor().getType());
  auto sizeInDim = tensorTy.getShape()[constDim.getValue().getZExtValue()];
  if (sizeInDim == ShapedType::kDynamic) {
    return nullptr;
  }
  return IntegerAttr::get(IntegerType::get(getContext(), 32), sizeInDim);
}
