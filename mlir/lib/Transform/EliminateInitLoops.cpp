#include "Transform/TTLPasses.h"
#include <llvm/Support/Debug.h>

namespace mlir {
namespace ttl {
#define GEN_PASS_DEF_TTLELIMINATEINITLOOPS
#include "Transform/TTLPasses.h.inc"

#define DEBUG_TYPE "ttl-eliminate-init-loops"

namespace {

class EliminateInitLoops
    : public impl::TTLEliminateInitLoopsBase<EliminateInitLoops> {
public:
  using impl::TTLEliminateInitLoopsBase<
      EliminateInitLoops>::TTLEliminateInitLoopsBase;

  void runOnOperation() final {

    func::FuncOp func = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Function name: " << func.getName() << "\n");
    func.walk([&](ttl::ForLoop forLoop) {
      if (eliminateLoopIfPossible(forLoop)) {
        ++numLoopsEliminated;
      }
    });
  }

private:
  bool eliminateLoopIfPossible(ttl::ForLoop forLoop) {
    // To eliminate an initialization loop, it must fulfill a number of
    // conditions:

    // 1. The loop must be directly in the function, i.e., not a nested loop or
    // inside an `if`.
    if (!isa<func::FuncOp>(forLoop->getParentOp())) {
      LLVM_DEBUG(llvm::dbgs() << "Failed condition 1\n");
      return false;
    }

    // 2. The loop's step must be a constant 1 and the start and end of the loop
    // must be constants.
    auto maybeConstantStep =
        dyn_cast_or_null<ttl::IntConstant>(forLoop.getStep().getDefiningOp());
    auto maybeConstantStart = dyn_cast_or_null<ttl::IntConstant>(
        forLoop.getLowerBound().getDefiningOp());
    auto maybeConstantEnd = dyn_cast_or_null<ttl::IntConstant>(
        forLoop.getUpperBound().getDefiningOp());
    if (!maybeConstantStep || maybeConstantStep.getConstVal() != 1 ||
        !maybeConstantStart || !maybeConstantEnd) {
      LLVM_DEBUG(llvm::dbgs() << "Failed condition 2\n");
      return false;
    }

    auto tensorSize =
        maybeConstantEnd.getConstVal() - maybeConstantStart.getConstVal();

    // 3. The loop must return a single result of a 1-dimensional tensor and
    // take a single init val of tensor type. The tensor must have static size
    // and must match the range between the constant start and end of the loop.
    if (forLoop.getNumResults() != 1 || forLoop.getInitArgs().size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "Failed condition 3-1\n");
      return false;
    }
    auto resultTy =
        dyn_cast<ttl::TensorType>(forLoop.getResults().front().getType());
    auto initArgTy =
        dyn_cast<ttl::TensorType>(forLoop.getInitArgs().front().getType());
    if (!resultTy || !initArgTy || resultTy.getShape().size() != 1 ||
        resultTy != initArgTy || resultTy.getShape().front() != tensorSize) {
      LLVM_DEBUG(llvm::dbgs() << "Failed condition 3-2\n");
      return false;
    }

    // 4. The init val of the loop must be a TensorEmpty which only has a single
    // user (the loop).
    auto maybeTensorEmpty = dyn_cast<ttl::TensorEmpty>(
        forLoop.getInitArgs().front().getDefiningOp());
    if (!maybeTensorEmpty || !maybeTensorEmpty.getTensor().hasOneUse()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed condition 4\n");
      return false;
    }

    // 5. The loop's body must consist of two operations, with the first one
    // being a TensorInsert (the other is the terminator/Yield) that takes a
    // single index.
    Block *bodyBlock = forLoop.getBody();
    if (bodyBlock->getOperations().size() != 2) {
      LLVM_DEBUG(llvm::dbgs() << "Failed condition 5-1\n");
      return false;
    }
    auto maybeTensorInsert = dyn_cast<ttl::TensorInsert>(bodyBlock->front());
    if (!maybeTensorInsert || maybeTensorInsert.getIndices().size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "Failed condition 5-2\n");
      return false;
    }

    // 6. The body block must have two arguments (one for the iteration
    // variable, one for the loop-carried tensor) and the tensor insert must use
    // them.
    if (bodyBlock->getNumArguments() != 2 ||
        bodyBlock->getArgument(0) != maybeTensorInsert.getIndices().front() ||
        bodyBlock->getArgument(0) != maybeTensorInsert.getValue() ||
        bodyBlock->getArgument(1) != maybeTensorInsert.getDest()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed condition 6\n");
      return false;
    }

    // Our candidate fulfills all conditions, so we are going to perform the
    // transformation:
    // 1. Replace the loop with a TensorRangeInit.
    // 2. Erase the original TensorEmpty.
    IRRewriter rewriter(forLoop);
    rewriter.replaceOpWithNewOp<ttl::TensorRangeInit>(
        forLoop, forLoop.getResultTypes().front(), forLoop.getLowerBound(),
        forLoop.getUpperBound());
    rewriter.eraseOp(maybeTensorEmpty);

    return true;
  }
};

} // namespace
} // namespace ttl
} // namespace mlir
