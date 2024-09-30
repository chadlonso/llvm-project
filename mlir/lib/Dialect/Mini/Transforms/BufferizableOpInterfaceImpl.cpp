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

struct AddOpBufferize
    : public BufferizableOpInterface::ExternalModel<AddOpBufferize,
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

    FailureOr<Value> maybeTrueBuffer =
        getBuffer(rewriter, selectOp.getLhs(), options);
    FailureOr<Value> maybeFalseBuffer =
        getBuffer(rewriter, selectOp.getRhs(), options);
    if (failed(maybeTrueBuffer) || failed(maybeFalseBuffer))
      return failure();
    Value trueBuffer = *maybeTrueBuffer;
    Value falseBuffer = *maybeFalseBuffer;

    replaceOpWithNewBufferizedOp<mini::AddOp>(
        rewriter, op, trueBuffer.getType(), trueBuffer, falseBuffer);
    return success();
  }
};

struct MatMulOpBufferize
    : public BufferizableOpInterface::ExternalModel<MatMulOpBufferize,
                                                    mini::MatMulOp> {
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
    auto selectOp = cast<mini::MatMulOp>(op);
    Location loc = selectOp.getLoc();

    FailureOr<Value> maybeTrueBuffer =
        getBuffer(rewriter, selectOp.getLhs(), options);
    FailureOr<Value> maybeFalseBuffer =
        getBuffer(rewriter, selectOp.getRhs(), options);
    if (failed(maybeTrueBuffer) || failed(maybeFalseBuffer))
      return failure();
    Value trueBuffer = *maybeTrueBuffer;
    Value falseBuffer = *maybeFalseBuffer;
    llvm::errs()<<"buffer type\n";
    //op->getResult(0).getType();
    replaceOpWithNewBufferizedOp<mini::MatMulOp>(
        rewriter, op, trueBuffer.getType(), trueBuffer, falseBuffer);
    return success();
  }
};

} // namespace

void mlir::mini::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, MiniDialect *dialect) {
    AddOp::attachInterface<AddOpBufferize>(*ctx);
    MatMulOp::attachInterface<MatMulOpBufferize>(*ctx);
  });
}
