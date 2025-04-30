#include "../TTLTypeConverter.h"
#include "Dialect/TTL/TTLDialect.h"
#include "Dialect/TTL/TTLOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
#define GEN_PASS_DEF_TTLTOSCALAR
#include "Conversion/Passes.h.inc"
} // namespace ttl
} // namespace mlir

using namespace mlir;
using namespace mlir::ttl;

#define DEBUG_TYPE "ttl-to-scalar"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace {

template <typename ToReplaceOp, typename IntReplaceOp,
          typename FloatReplaceOp = void>
struct BinaryOpLowering : public OpConversionPattern<ToReplaceOp> {

  using OpConversionPattern<ToReplaceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ToReplaceOp op, typename ToReplaceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = this->typeConverter->convertType(op.getResult().getType());
    if (!resultTy.isIntOrFloat()) {
      // Tensor binary operations are handled by a different pass.
      return failure();
    }
    if (resultTy.isInteger()) {
      rewriter.replaceOpWithNewOp<IntReplaceOp>(op, adaptor.getLeft(),
                                                adaptor.getRight());
      return success();
    }
    if constexpr (!std::is_void_v<FloatReplaceOp>) {
      rewriter.replaceOpWithNewOp<FloatReplaceOp>(op, adaptor.getLeft(),
                                                  adaptor.getRight());
      return success();
    }
    return failure();
  }
};

struct NotLowering : public OpConversionPattern<ttl::Not> {
  using OpConversionPattern<ttl::Not>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::Not op, ttl::Not::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Arith dialect does not define a bitwise not right now. So we use a trick
    // here to get our bitwise not: We XOR the value with an all-1 integer
    // value (-1 in two's complement).
    Value allOnes = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(-1));
    rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, adaptor.getOp(), allOnes);
    return success();
  }
};

struct MinusLowering : public OpConversionPattern<ttl::Minus> {

  using OpConversionPattern<ttl::Minus>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::Minus op, ttl::Minus::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resulTy = typeConverter->convertType(op.getResult().getType());
    if (resulTy.isInteger()) {
      // Arith dialect does not define a negate/minus operation for integer
      // right now. So we simply multiply with -1.
      Value minusOne = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getI32IntegerAttr(-1));
      rewriter.replaceOpWithNewOp<arith::MulIOp>(op, adaptor.getOp(), minusOne);
      return success();
    }
    rewriter.replaceOpWithNewOp<arith::NegFOp>(op, adaptor.getOp());
    return success();
  }
};

struct IntegerCompareLowering : OpConversionPattern<ttl::Compare> {
  using OpConversionPattern<ttl::Compare>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::Compare op, ttl::Compare::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getLeft().getType().isInteger()) {
      return failure();
    }
    arith::CmpIPredicate pred = [](ttl::TTLCmpOpcodes code) {
      switch (code) {
      case mlir::ttl::TTLCmpOpcodes::EQ:
        return arith::CmpIPredicate::eq;
      case mlir::ttl::TTLCmpOpcodes::NE:
        return arith::CmpIPredicate::ne;
      case mlir::ttl::TTLCmpOpcodes::GE:
        return arith::CmpIPredicate::sge;
      case mlir::ttl::TTLCmpOpcodes::GT:
        return arith::CmpIPredicate::sgt;
      case mlir::ttl::TTLCmpOpcodes::LT:
        return arith::CmpIPredicate::slt;
      case mlir::ttl::TTLCmpOpcodes::LE:
        return arith::CmpIPredicate::sle;
      }
    }(op.getOpcode());
    auto cmp = rewriter.create<arith::CmpIOp>(
        op.getLoc(), pred, adaptor.getLeft(), adaptor.getRight());
    rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, rewriter.getI32Type(), cmp);
    return success();
  }
};

struct FloatCompareLowering : OpConversionPattern<ttl::Compare> {
  using OpConversionPattern<ttl::Compare>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::Compare op, ttl::Compare::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getLeft().getType().isF32()) {
      return failure();
    }
    arith::CmpFPredicate pred = [](ttl::TTLCmpOpcodes code) {
      switch (code) {
      case mlir::ttl::TTLCmpOpcodes::EQ:
        return arith::CmpFPredicate::UEQ;
      case mlir::ttl::TTLCmpOpcodes::NE:
        return arith::CmpFPredicate::UNE;
      case mlir::ttl::TTLCmpOpcodes::GE:
        return arith::CmpFPredicate::UGE;
      case mlir::ttl::TTLCmpOpcodes::GT:
        return arith::CmpFPredicate::UGT;
      case mlir::ttl::TTLCmpOpcodes::LT:
        return arith::CmpFPredicate::ULT;
      case mlir::ttl::TTLCmpOpcodes::LE:
        return arith::CmpFPredicate::ULE;
      }
    }(op.getOpcode());
    auto cmp = rewriter.create<arith::CmpFOp>(
        op.getLoc(), pred, adaptor.getLeft(), adaptor.getRight());
    rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, rewriter.getI32Type(), cmp);
    return success();
  }
};

struct IfLowering : OpConversionPattern<ttl::If> {
  using OpConversionPattern<ttl::If>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::If op, ttl::If::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTys;
    for (auto t : op.getResultTypes()) {
      resultTys.push_back(typeConverter->convertType(t));
    }
    Value newCond = adaptor.getCond();
    Value zero = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getIntegerAttr(newCond.getType(), 0));
    auto nonZero = rewriter.create<arith::CmpIOp>(
        op.getLoc(), arith::CmpIPredicate::ne, newCond, zero);
    auto newIf = rewriter.create<scf::IfOp>(op.getLoc(), resultTys, nonZero);
    newIf.getThenRegion().takeBody(op.getThenRegion());
    newIf.getElseRegion().takeBody(op.getElseRegion());
    rewriter.replaceOp(op, newIf);
    return success();
  }
};

struct ForLoopConversion : OpConversionPattern<ttl::ForLoop> {
  using OpConversionPattern<ttl::ForLoop>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::ForLoop op, ttl::ForLoop::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    scf::ForOp newLoop = rewriter.create<scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), adaptor.getInitArgs());

    if (failed(rewriter.convertRegionTypes(&op.getRegion(), *typeConverter))) {
      return rewriter.notifyMatchFailure(op, "region type conversion failed");
    }
    rewriter.mergeBlocks(op.getBody(0), newLoop.getBody(0),
                         newLoop.getBody(0)->getArguments());
    rewriter.replaceOp(op, newLoop);
    return success();
  }
};

struct YieldLowering : OpConversionPattern<ttl::Yield> {
  using OpConversionPattern<ttl::Yield>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::Yield op, ttl::Yield::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

struct FuncConversion : OpConversionPattern<func::FuncOp> {

  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, func::FuncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> argTys;
    FunctionType oldFuncTy = op.getFunctionType();
    for (auto t : oldFuncTy.getInputs()) {
      argTys.push_back(typeConverter->convertType(t));
    }
    FunctionType newFuncTy = rewriter.getFunctionType(
        argTys, typeConverter->convertType(oldFuncTy.getResult(0)));
    auto newFunc =
        rewriter.create<func::FuncOp>(op.getLoc(), op.getName(), newFuncTy);
    rewriter.inlineRegionBefore(op.getFunctionBody(), newFunc.getBody(),
                                newFunc.end());
    if (failed(
            rewriter.convertRegionTypes(&newFunc.getBody(), *typeConverter))) {
      return rewriter.notifyMatchFailure(op, "region type conversion failed");
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct CallConverion : OpConversionPattern<func::CallOp> {
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, func::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> NewResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getResultTypes(), NewResultTypes))) {
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    }
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, NewResultTypes, adaptor.getCallee(), adaptor.getOperands());
    return success();
  }
};

struct ReturnLowering : OpConversionPattern<ttl::Return> {
  using OpConversionPattern<ttl::Return>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttl::Return op, ttl::Return::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

template <typename ReplaceOp>
struct ConstantLowering : OpConversionPattern<ReplaceOp> {
  using OpConversionPattern<ReplaceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReplaceOp op, typename ReplaceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getConstValAttr());
    return success();
  }
};

struct ConvertTTLToScalarPass
    : public ttl::impl::TTLToScalarBase<ConvertTTLToScalarPass> {

  using ttl::impl::TTLToScalarBase<ConvertTTLToScalarPass>::TTLToScalarBase;

  void runOnOperation() override;
};

} // namespace

void ConvertTTLToScalarPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<tensor::TensorDialect, scf::SCFDialect, BuiltinDialect,
                         func::FuncDialect, index::IndexDialect,
                         arith::ArithDialect, linalg::LinalgDialect>();

  target.addLegalDialect<ttl::TTLDialect>();
  target.addDynamicallyLegalOp<ttl::Add, ttl::Sub, ttl::Mul, ttl::Div>(
      [](Operation *op) {
        return isa<ttl::TensorType>(op->getResults().front().getType());
      });
  TTLTypeConverter typeConverter;
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });
  target.addIllegalOp<ttl::And, ttl::Or, ttl::Not, ttl::Minus, ttl::Compare,
                      ttl::If, ttl::Yield, ttl::Return, ttl::FloatConstant,
                      ttl::IntConstant, ttl::ForLoop>();

  RewritePatternSet patterns(&getContext());
  patterns
      .add<BinaryOpLowering<ttl::Add, arith::AddIOp, arith::AddFOp>,
           BinaryOpLowering<ttl::Sub, arith::SubIOp, arith::SubFOp>,
           BinaryOpLowering<ttl::Mul, arith::MulIOp, arith::MulFOp>,
           BinaryOpLowering<ttl::Div, arith::DivSIOp, arith::DivFOp>,
           BinaryOpLowering<ttl::And, arith::AndIOp>,
           BinaryOpLowering<ttl::Or, arith::OrIOp>, NotLowering, MinusLowering,
           IntegerCompareLowering, FloatCompareLowering, IfLowering,
           YieldLowering, FuncConversion, CallConverion, ReturnLowering,
           ForLoopConversion, ConstantLowering<ttl::FloatConstant>,
           ConstantLowering<ttl::IntConstant>>(typeConverter, &getContext());

  ModuleOp Module = getOperation();

  if (failed(applyPartialConversion(Module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}
