#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Passes/TTLPasses.h"

namespace mlir {
namespace ttl {
#define GEN_PASS_DEF_TTLTILING
#include "Passes/TTLPasses.h.inc"

namespace {

class TTLTilingPattern : public OpRewritePattern<mlir::ttl::MatMul> {
public:
  using OpRewritePattern<mlir::ttl::MatMul>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ttl::MatMul op,
                                PatternRewriter &rewriter) const final {
    scf::SCFTilingOptions tilingOptions;
    SmallVector<OpFoldResult> tilingSizes;
    tilingSizes.push_back(rewriter.getIndexAttr(8));
    tilingSizes.push_back(rewriter.getIndexAttr(8));
    tilingOptions.setTileSizes(tilingSizes);
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

    FailureOr<scf::SCFTilingResult> tiledResults =
        scf::tileUsingSCF(rewriter, op, tilingOptions);

    if (failed(tiledResults)) {
      return failure();
    }
    rewriter.replaceOp(op, tiledResults->mergeResult.replacements);
    return success();
  }
};

class TTLTiling : public impl::TTLTilingBase<TTLTiling> {
public:
  using impl::TTLTilingBase<TTLTiling>::TTLTilingBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTLTilingPattern>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace ttl
} // namespace mlir
