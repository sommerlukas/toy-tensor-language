#include "Dialect/TTL/TTLOps.h"
#include "Dialect/TTL/TTLAttributes.h"
#include "Dialect/TTL/TTLDialect.h"
#include "Dialect/TTL/TTLTypes.h"
#include "mlir/IR/OpImplementation.h"
#include <llvm/Support/Debug.h>

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

LogicalResult MatMul::verify() {
  auto leftTTy = dyn_cast<ttl::TensorType>(getLeft().getType());
  auto rightTTy = dyn_cast<ttl::TensorType>(getRight().getType());
  auto resTTy = dyn_cast<ttl::TensorType>(getType());
  if (!leftTTy || !rightTTy || !resTTy ||
      leftTTy.getElementType() != rightTTy.getElementType() ||
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

  // Allowed: ? -> ?, 4 -> 4, 4 -> ?, but not: 4 -> 5, ? -> 4
  if (!(leftShape[0] == resShape[0] || ShapedType::isDynamic(resShape[0])))
    return emitError("result shape mismatch in first dimension");

  if (!(rightShape[1] == resShape[1] || ShapedType::isDynamic(resShape[1])))
    return emitError("result shape mismatch in second dimension");

  return success();
}

SmallVector<Range> MatMul::getIterationDomain(OpBuilder &builder) {
  auto Loc = getLoc();
  auto getSize = [&](Value tensorOp, unsigned index) -> OpFoldResult {
    auto opTy = cast<ttl::TensorType>(tensorOp.getType());
    auto opShape = opTy.getShape();
    if (!ShapedType::isDynamic(opShape[index])) {
      return builder.getIndexAttr(opShape[index]);
    } else {
      auto indexCst = builder.create<mlir::ttl::IntConstant>(Loc, index);
      auto dim = builder.create<mlir::ttl::Dim>(Loc, tensorOp, indexCst);
      return builder
          .create<mlir::ttl::IndexCast>(Loc, builder.getIndexType(), dim)
          .getOutput();
    }
  };
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);

  SmallVector<Range> loopBounds(3);
  // M
  loopBounds[0].offset = zero;
  loopBounds[0].size = getSize(getLeft(), 0);
  loopBounds[0].stride = one;
  // N
  loopBounds[1].offset = zero;
  loopBounds[1].size = getSize(getRight(), 1);
  loopBounds[1].stride = one;
  // K
  loopBounds[2].offset = zero;
  loopBounds[2].size = getSize(getLeft(), 1);
  loopBounds[2].stride = one;

  llvm::dbgs() << "Returning iteration domain\n";

  return loopBounds;
}

SmallVector<utils::IteratorType> MatMul::getLoopIteratorTypes() {
  return SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                          utils::IteratorType::parallel,
                                          utils::IteratorType::reduction};
}

LogicalResult MatMul::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  llvm::dbgs() << "Result number: " << resultNumber << "\n";
  llvm::dbgs() << "Offsets:\n";
  for (const auto &off : offsets) {
    off.dump();
  }
  llvm::dbgs() << "Sizes:\n";
  for (const auto &s : sizes) {
    s.dump();
  }
  if(resultNumber != 0){
    return failure();
  }

  resultOffsets.push_back(offsets[0]);
  resultOffsets.push_back(offsets[1]);
  resultSizes.push_back(sizes[0]);
  resultSizes.push_back(sizes[1]);
  return success();
}

FailureOr<TilingResult>
MatMul::getTiledImplementation(OpBuilder &builder,
                               ArrayRef<OpFoldResult> offsets,
                               ArrayRef<OpFoldResult> sizes) {
  llvm::dbgs() << "Offsets:\n";
  for (const auto &off : offsets) {
    off.dump();
  }
  llvm::dbgs() << "Sizes:\n";
  for (const auto &s : sizes) {
    s.dump();
  }
  auto Loc = getLoc();
  auto TTLIntTy = mlir::ttl::IntType::get(getContext());
  auto toTTLInt = [&](OpFoldResult idx) -> Value {
    if (auto IdxVal = dyn_cast<Value>(idx)) {
      return builder.create<mlir::ttl::IndexCast>(Loc, TTLIntTy, IdxVal)
          .getOutput();
    }
    auto RawAttr = cast<mlir::Attribute>(idx);
    auto IdxAttr = cast<mlir::IntegerAttr>(RawAttr);
    return builder.create<mlir::ttl::IntConstant>(
        Loc, IdxAttr.getValue().getZExtValue());
  };

  auto elemTy =
      cast<mlir::ttl::TensorType>(getLeft().getType()).getElementType();

  auto extractTensor = [&](Value tensor, unsigned idx1,
                           unsigned idx2) -> Value {
    SmallVector<Value> sliceOffsets;
    sliceOffsets.push_back(toTTLInt(offsets[idx1]));
    sliceOffsets.push_back(toTTLInt(offsets[idx2]));
    SmallVector<Value> sliceSizes;
    sliceSizes.push_back(toTTLInt(sizes[idx1]));
    sliceSizes.push_back(toTTLInt(sizes[idx2]));
    auto sliceTy = mlir::ttl::TensorType::get(
        elemTy, {ShapedType::kDynamic, ShapedType::kDynamic});
    return builder
        .create<mlir::ttl::Slice>(Loc, sliceTy, tensor, sliceOffsets,
                                  sliceSizes)
        .getResult();
  };

  auto leftSlice = extractTensor(getLeft(), 0, 2);
  auto rightSlice = extractTensor(getRight(), 2, 1);

  auto getSize = [&](OpFoldResult size) -> int64_t {
    if (isa<Value>(size)) {
      return ShapedType::kDynamic;
    }
    auto rawAttr = cast<mlir::Attribute>(size);
    auto attr = cast<mlir::IntegerAttr>(rawAttr);
    return attr.getValue().getZExtValue();
    return 0;
  };
  auto tileTy = mlir::ttl::TensorType::get(
      elemTy, {getSize(sizes[0]), getSize(sizes[1])});
  Operation *matmulTile =
      builder.create<mlir::ttl::MatMul>(Loc, tileTy, leftSlice, rightSlice);

  return TilingResult{{matmulTile},
                      {matmulTile->getResult(0)},
                      {leftSlice.getDefiningOp(), rightSlice.getDefiningOp()}};
}

// https://github.com/llvm/llvm-project/blob/main/mlir/test/Interfaces/TilingInterface/tile-using-interface.mlir
