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

static bool isCompatibleSize(int64_t aSz, int64_t bSz) {
  // Allowed: ? -> ?, 4 -> 4, ? -> 4, 4 -> ?, but not: 4 -> 5
  return aSz == bSz || ShapedType::isDynamic(aSz) || ShapedType::isDynamic(bSz);
}

static bool isCompatible(Type aTy, Type bTy) {
  // Check trivial situation first.
  if (aTy == bTy)
    return true;

  auto aTTy = dyn_cast<ttl::TensorType>(aTy);
  auto bTTy = dyn_cast<ttl::TensorType>(bTy);

  // Only continue if both types are tensors. If not, we either have different
  // scalar types, or a combination of tensor and scalar.
  if (!(aTTy && bTTy))
    return false;

  if (aTTy.getElementType() != bTTy.getElementType())
    return false;

  auto aSh = aTTy.getShape();
  auto bSh = bTTy.getShape();
  if (aSh.size() != bSh.size())
    return false;

  return llvm::all_of_zip(aSh, bSh, isCompatibleSize);
}

LogicalResult Return::verify() {
  auto function = cast<func::FuncOp>((*this)->getParentOp());
  auto functionResultTypes = function.getFunctionType().getResults();

  if (functionResultTypes.size() != 1)
    return emitError("enclosing function @")
           << function.getSymName() << " expects " << functionResultTypes.size()
           << " results";

  if (!isCompatible(getRetVal().getType(), functionResultTypes.front()))
    return emitError("operand is a ")
           << getRetVal().getType() << ", but the enclosing function @"
           << function.getSymName() << " expects "
           << functionResultTypes.front();

  return success();
}

LogicalResult MatMul::verify() {
  auto leftTTy = cast<ttl::TensorType>(getLeft().getType());
  auto rightTTy = cast<ttl::TensorType>(getRight().getType());
  auto resTTy = cast<ttl::TensorType>(getType());
  if (leftTTy.getElementType() != rightTTy.getElementType() ||
      rightTTy.getElementType() != resTTy.getElementType())
    return emitError(
        "operands and result must be tensors of the same element type");

  auto leftShape = leftTTy.getShape();
  auto rightShape = rightTTy.getShape();
  auto resShape = resTTy.getShape();

  if (leftShape.size() != 2 || rightShape.size() != 2 || resShape.size() != 2)
    return emitError("operand- and result tensors must be 2-dimensional");

  if (!ShapedType::isDynamic(leftShape[1]) &&
      !ShapedType::isDynamic(rightShape[0]) && leftShape[1] != rightShape[0])
    return emitError("shape mismatch in common dimension");

  if (!isCompatibleSize(leftShape[0], resShape[0]))
    return emitError("result shape mismatch in first dimension");

  if (!isCompatibleSize(rightShape[1], resShape[1]))
    return emitError("result shape mismatch in second dimension");

  return success();
}

LogicalResult ttl::verifyBinOp(Operation *op) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return op->emitOpError("is not a binary op");

  Value left, right, res;
  left = op->getOperand(0);
  right = op->getOperand(1);
  res = op->getResult(0);

  Type leftTy = left.getType();
  Type rightTy = right.getType();
  Type resTy = res.getType();

  bool leftAndRight = isCompatible(leftTy, rightTy);
  bool leftAndRes = isCompatible(leftTy, resTy);
  bool rightAndRes = isCompatible(rightTy, resTy);

  auto leftTTy = dyn_cast<ttl::TensorType>(leftTy);
  auto rightTTy = dyn_cast<ttl::TensorType>(rightTy);

  // Scalar or elementwise operation
  if (!(static_cast<bool>(leftTTy) ^ static_cast<bool>(rightTTy))) {
    if (!(leftAndRight && leftAndRes && rightAndRes))
      return op->emitError("incompatible operand and result types");
    return success();
  }

  // Tensor-scalar (or vice versa)
  if ((leftTTy && leftTTy.getElementType() != rightTy) ||
      (rightTTy && rightTTy.getElementType() != leftTy))
    return op->emitError(
        "scalar operand's type does not match tensor element type");

  if ((leftTTy && !leftAndRes) || (rightTTy && !rightAndRes))
    return op->emitError(
        "tensor operand's type cannot be assigned to the result type");

  return success();
}

SmallVector<Region *> ForLoop::getLoopRegions() { return {&getBodyRegion()}; }

std::optional<SmallVector<Value>> ForLoop::getLoopInductionVars() {
  return SmallVector<Value>{getBody()->getArgument(0)};
}

std::optional<SmallVector<OpFoldResult>> ForLoop::getLoopLowerBounds() {
  return SmallVector<OpFoldResult>{getLowerBound()};
}

std::optional<SmallVector<OpFoldResult>> ForLoop::getLoopUpperBounds() {
  return SmallVector<OpFoldResult>{getUpperBound()};
}

std::optional<SmallVector<OpFoldResult>> ForLoop::getLoopSteps() {
  return SmallVector<OpFoldResult>{getStep()};
}

Block::BlockArgListType ForLoop::getRegionIterArgs() {
  return getBody()->getArguments().drop_front(1);
}

MutableArrayRef<OpOperand> ForLoop::getInitsMutable() {
  return getInitArgsMutable();
}

std::optional<MutableArrayRef<OpOperand>> ForLoop::getYieldedValuesMutable() {
  return cast<ttl::Yield>(getBody()->getTerminator()).getResultsMutable();
}

std::optional<ResultRange> ForLoop::getLoopResults() { return getResults(); }
