#include "mlir/Dialect/Mini/Transforms/Passes.h"

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include <optional>

namespace mlir {
namespace mini {
#define GEN_PASS_DEF_MINITILING
#include "mlir/Dialect/Mini/Transforms/Passes.h.inc"
} // namespace mini
} // namespace mlir

#define DEBUG_TYPE "tile-mini-ops"

using namespace mlir;
using namespace mlir::mini;

namespace {
/// Loop unroll jam pass. Currently, this just unroll jams the first
/// outer loop in a Function.
struct Tiling
    : public mini::impl::MiniTilingBase<Tiling> {
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::mini::createTilingPass() {
  return std::make_unique<Tiling>();
}

void Tiling::runOnOperation() {
  SmallVector<PartialReductionOpInterface, 4> opsList;
  getOperation()->walk([&](mlir::Operation *op) {
    if(auto interFace = dyn_cast<PartialReductionOpInterface>(op)) {
      opsList.push_back(interFace);
    }
  });
  for(auto iFace : opsList) 
  {
      OpBuilder builder(iFace);
      IRRewriter rewriter(&getContext());
      scf::SCFTilingOptions tilingOptions;
      auto constEight = builder.create<arith::ConstantIndexOp>(iFace->getLoc(), 8).getValue();
      SmallVector<OpFoldResult,4> tileSize;
      tileSize.push_back(constEight);
      tileSize.push_back(constEight);
      tileSize.push_back(constEight);
      tilingOptions.setTileSizes(tileSize);
      auto res = scf::tileReductionUsingScf(rewriter, iFace, tileSize);
      iFace->replaceAllUsesWith(res->replacements);
      iFace->erase();
  }
}
