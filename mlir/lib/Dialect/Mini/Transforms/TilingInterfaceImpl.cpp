//===- MiniTilingInterface.cpp - Tiling Interface  models *- C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mini/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Mini/IR/Mini.h"
//#include "mlir/Dialect/Mini/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::mini;

namespace {

struct AddOpTiling : public TilingInterface::ExternalModel<AddOpTiling, AddOp> {

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto padOp = cast<AddOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        2/*rank*/, utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    ReifiedRankedShapedTypeDims reifiedShapes;
    (void)reifyResultShapes(b, op, reifiedShapes);
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    // Initialize all the ranges to {zero, one, one}. All the `ub`s are
    // overwritten.
    SmallVector<Range> loopRanges(reifiedShapes[0].size(), {zero, one, one});
    for (const auto &ub : enumerate(reifiedShapes[0]))
      loopRanges[ub.index()].size = ub.value();
    return loopRanges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    FailureOr<TilingResult> result =
        mini::bubbleUpPadSlice(b, cast<AddOp>(op), offsets, sizes);
    if (failed(result))
      return failure();
    return result.value();
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
};
}

void mlir::mini::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, MiniDialect *dialect) {
    mini::AddOp::attachInterface<AddOpTiling>(*ctx);
  });
}
