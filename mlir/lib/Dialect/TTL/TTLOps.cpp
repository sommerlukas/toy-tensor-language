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

OpFoldResult Slice::fold(FoldAdaptor adaptor) {
  if (!getTensor().getDefiningOp()) {
    return nullptr;
  }
  auto scalarFill =
      dyn_cast<ttl::TensorScalarInit>(getTensor().getDefiningOp());
  if (!scalarFill) {
    return nullptr;
  }
  for (auto S : adaptor.getSizes()) {
    if (!S) {
      // Null attribute, one of the sizes wasn't constant
      return nullptr;
    }

    IntegerAttr constSize = dyn_cast<IntegerAttr>(S);
    if (!constSize) {
      return nullptr;
    }
    if (constSize.getValue().getZExtValue() != 1) {
      return nullptr;
    }
  }
  return scalarFill.getInitVal();
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

namespace {

struct ListInitCanonicalizer : public OpRewritePattern<ttl::TensorListInit> {
  using OpRewritePattern<ttl::TensorListInit>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttl::TensorListInit op,
                                PatternRewriter &rewriter) const override {
    Value firstElement = op.getElements().front();
    bool allEqual = llvm::all_of(
        op.getElements(), [&](Value elem) { return elem == firstElement; });
    if (!allEqual) {
      return rewriter.notifyMatchFailure(op, "Not all elements equal");
    }

    rewriter.replaceOpWithNewOp<ttl::TensorScalarInit>(
        op, op.getResult().getType(), firstElement);
    return success();
  }
};
} // namespace

void TensorListInit::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add(std::make_unique<ListInitCanonicalizer>(context));
}

LogicalResult Return::verify() {
  auto function = cast<func::FuncOp>((*this)->getParentOp());
  auto functionResultTypes = function.getFunctionType().getResults();

  if (functionResultTypes.size() != 1)
    return emitError("enclosing function @")
           << function.getSymName() << " expects " << functionResultTypes.size()
           << " results";

  if (getRetVal().getType() != functionResultTypes.front())
    return emitError("operand is a ")
           << getRetVal().getType() << ", but the enclosing function @"
           << function.getSymName() << " expects "
           << functionResultTypes.front();

  return success();
}
