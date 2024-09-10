//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mini/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Mini/IR/Mini.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {

/// Bufferization of arith.select. Just replace the operands.
struct SelectOpInterface
    : public BufferizableOpInterface::ExternalModel<SelectOpInterface,
                                                    mini::AddOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getOpResult(0) /*result*/, BufferRelation::Equivalent,
             /*isDefinite=*/false}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto selectOp = cast<mini::AddOp>(op);
    Location loc = selectOp.getLoc();

    // TODO: It would be more efficient to copy the result of the `select` op
    // instead of its OpOperands. In the worst case, 2 copies are inserted at
    // the moment (one for each tensor). When copying the op result, only one
    // copy would be needed.
    FailureOr<Value> maybeTrueBuffer =
        getBuffer(rewriter, selectOp.getLhs(), options);
    FailureOr<Value> maybeFalseBuffer =
        getBuffer(rewriter, selectOp.getRhs(), options);
    if (failed(maybeTrueBuffer) || failed(maybeFalseBuffer))
      return failure();
    Value trueBuffer = *maybeTrueBuffer;
    Value falseBuffer = *maybeFalseBuffer;

    replaceOpWithNewBufferizedOp<mini::AddOp>(
        rewriter, op, trueBuffer, falseBuffer);
    return success();
  }
};

} // namespace

void mlir::mini::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, MiniDialect *dialect) {
    AddOp::attachInterface<SelectOpInterface>(*ctx);
  });
}
