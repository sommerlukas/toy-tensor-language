#include "Conversion/TTLToTensor/TTLToTensor.h"

#include "Dialect/TTL/TTLDialect.h"
#include "Dialect/TTL/TTLOps.h"
#include "TTLTypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
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
#define GEN_PASS_DEF_TTLTOTENSOR
#include "Conversion/Passes.h.inc"
} // namespace ttl
} // namespace mlir

using namespace mlir;
using namespace mlir::ttl;

#define DEBUG_TYPE "ttl-to-tensor"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace {

struct ScalarInitLowering : public OpConversionPattern<ttl::TensorScalarInit> {

  using OpConversionPattern<ttl::TensorScalarInit>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::TensorScalarInit op,
                  ttl::TensorScalarInit::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newType = typeConverter->convertType(op.getTensor().getType());
    rewriter.replaceOpWithNewOp<tensor::SplatOp>(op, newType,
                                                 adaptor.getInitVal());
    return success();
  }
};

struct ListInitLowering : public OpConversionPattern<ttl::TensorListInit> {
  using OpConversionPattern<ttl::TensorListInit>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::TensorListInit op, ttl::TensorListInit::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getOperands());
    return success();
  }
};

struct RangeInitLowering : public OpConversionPattern<ttl::TensorRangeInit> {
  using OpConversionPattern<ttl::TensorRangeInit>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::TensorRangeInit op,
                  ttl::TensorRangeInit::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(adaptor.getOperands().size() == 2);
    Value start = adaptor.getStart();
    Value end = adaptor.getEnd();
    auto tensorTy = llvm::cast<mlir::RankedTensorType>(
        this->typeConverter->convertType(op->getResults().front().getType()));
    auto Loc = op->getLoc();
    SmallVector<Value> dynamicExtents;
    if (!tensorTy.hasStaticShape()) {
      Value extentStart =
          rewriter.create<index::CastSOp>(Loc, rewriter.getIndexType(), start);
      Value extentEnd =
          rewriter.create<index::CastSOp>(Loc, rewriter.getIndexType(), end);
      Value extent = rewriter.create<index::SubOp>(Loc, extentEnd, extentStart);
      dynamicExtents.push_back(extent);
    }
    auto generateOp =
        rewriter.create<tensor::GenerateOp>(Loc, tensorTy, dynamicExtents);
    Block *generateBlock = rewriter.createBlock(&generateOp.getBody());
    for (int64_t i = 0; i < tensorTy.getRank(); ++i) {
      generateBlock->addArgument(rewriter.getIndexType(), Loc);
    }
    rewriter.setInsertionPointToStart(generateBlock);
    Value index;
    int i = 0;
    for (; i < tensorTy.getRank() - 1; ++i) {
      auto genIndex = generateBlock->getArgument(i);
      Value tensorSize = rewriter.create<index::ConstantOp>(
          Loc, rewriter.getIndexAttr(tensorTy.getShape()[i]));
      Value mul = rewriter.create<index::MulOp>(Loc, genIndex, tensorSize);
      if (index) {
        index = rewriter.create<index::AddOp>(Loc, index, mul);
      } else {
        index = mul;
      }
    }
    if (index) {
      index = rewriter.create<index::AddOp>(Loc, index,
                                            generateBlock->getArgument(i));
    } else {
      index = generateBlock->getArgument(i);
    }
    Value elem =
        rewriter.create<index::CastSOp>(Loc, rewriter.getI32Type(), index);
    elem = rewriter.create<arith::AddIOp>(Loc, start, elem);
    rewriter.create<tensor::YieldOp>(Loc, elem);
    rewriter.replaceOp(op, generateOp);
    return success();
  }
};

struct InsertLowering : public OpConversionPattern<ttl::TensorInsert> {
  using OpConversionPattern<ttl::TensorInsert>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::TensorInsert op, ttl::TensorInsert::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location Loc = op.getLoc();
    SmallVector<Value> indices;
    for (Value I : adaptor.getIndices()) {
      indices.push_back(
          rewriter.create<index::CastSOp>(Loc, rewriter.getIndexType(), I));
    }
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, adaptor.getValue(),
                                                  adaptor.getDest(), indices);
    return success();
  }
};

struct DimLowering : public OpConversionPattern<ttl::Dim> {

  using OpConversionPattern<ttl ::Dim>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::Dim op, ttl::Dim::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value idx = rewriter.create<index::CastSOp>(
        op.getLoc(), rewriter.getIndexType(), adaptor.getDimension());
    Value dim =
        rewriter.create<tensor::DimOp>(op.getLoc(), adaptor.getTensor(), idx);
    rewriter.replaceOpWithNewOp<index::CastSOp>(op, rewriter.getI32Type(), dim);
    return success();
  }
};

struct SliceLowering : public OpConversionPattern<ttl::Slice> {
  using OpConversionPattern<Slice>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::Slice op, ttl::Slice::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto tensorTy = dyn_cast<RankedTensorType>(
        typeConverter->convertType(op.getResult().getType()));
    if (!tensorTy) {
      return failure();
    }
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    for (int64_t i = 0; i < tensorTy.getRank(); ++i) {
      Value offset = rewriter.create<index::CastSOp>(
          op.getLoc(), rewriter.getIndexType(), adaptor.getOffsets()[i]);
      offsets.push_back(offset);
      if (tensorTy.isDynamicDim(i)) {
        Value size = rewriter.create<index::CastSOp>(
            op.getLoc(), rewriter.getIndexType(), adaptor.getSizes()[i]);
        sizes.push_back(size);
      } else {
        sizes.push_back(rewriter.getIndexAttr(tensorTy.getDimSize(i)));
      }
      strides.push_back(rewriter.getIndexAttr(1));
    }
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, tensorTy, adaptor.getTensor(), offsets, sizes, strides);
    return success();
  }
};

struct SliceSingleLowering : public OpConversionPattern<ttl::Slice> {
  using OpConversionPattern<Slice>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::Slice op, ttl::Slice::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resulTy = typeConverter->convertType(op.getResult().getType());
    if (!resulTy.isIntOrFloat()) {
      return failure();
    }
    SmallVector<Value> indices;
    for (Value I : adaptor.getOffsets()) {
      indices.push_back(rewriter.create<index::CastSOp>(
          op.getLoc(), rewriter.getIndexType(), I));
    }
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
        op, resulTy, adaptor.getTensor(), indices);
    return success();
  }
};

struct ConvertTTLToTensorPass
    : public ttl::impl::TTLToTensorBase<ConvertTTLToTensorPass> {

  using ttl::impl::TTLToTensorBase<ConvertTTLToTensorPass>::TTLToTensorBase;

  void runOnOperation() override;
};

} // namespace

void ConvertTTLToTensorPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<tensor::TensorDialect, scf::SCFDialect, BuiltinDialect,
                         func::FuncDialect, index::IndexDialect,
                         arith::ArithDialect>();

  target.addLegalDialect<ttl::TTLDialect>();
  target.addIllegalOp<ttl::TensorScalarInit, ttl::TensorRangeInit,
                      ttl::TensorListInit, ttl::TensorInsert, ttl::Dim,
                      ttl::Slice>();

  TTLTypeConverter typeConverter;

  RewritePatternSet patterns(&getContext());
  patterns.add<ScalarInitLowering, RangeInitLowering, ListInitLowering,
               InsertLowering, DimLowering, SliceLowering, SliceSingleLowering>(
      typeConverter, &getContext());

  ModuleOp Module = getOperation();

  if (failed(applyPartialConversion(Module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}
