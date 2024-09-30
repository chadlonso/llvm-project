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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Mini/IR/Mini.h"
//#include "mlir/Dialect/Mini/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::mini;

namespace {

struct AddOpTiling : public TilingInterface::ExternalModel<AddOpTiling, AddOp> {
  static Value getSlice(OpBuilder &b, Location loc, Value source,
                      ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides) {
    return TypeSwitch<Type, Value>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        return b.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                           strides);
      })
      .Default([&](Type t) { return nullptr; });
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto padOp = cast<AddOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        2/*rank*/, utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto tnsrTy = op->getResult(0).getType().dyn_cast<TensorType>();
    auto shape = tnsrTy.getShape();
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    // Initialize all the ranges to {zero, one, one}. All the `ub`s are
    // overwritten.
    SmallVector<Range> loopRanges(2, {zero, one, one});
    int i =0;
    for (const auto &ub : shape)
      loopRanges[i++].size = b.getIndexAttr(ub);
    return loopRanges;
  }

  static FlatSymbolRefAttr getFunc(ModuleOp module, StringRef name,
                                               TypeRange resultType,
                                               ValueRange operands
                                               ) {
  MLIRContext *context = module.getContext();
  auto result = SymbolRefAttr::get(context, name);
  auto func = module.lookupSymbol<mlir::func::FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<mlir::func::FuncOp>(
        module.getLoc(), name,
        FunctionType::get(context, operands.getTypes(), resultType));
    func.setPrivate();
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(context));
  }
  return result;
}

static func::CallOp createFuncCall(
    OpBuilder &builder, Location loc, StringRef name, TypeRange resultType,
    ValueRange operands) {
  auto module = builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  FlatSymbolRefAttr fn =
      getFunc(module, name, resultType, operands);
  return builder.create<mlir::func::CallOp>(loc, resultType, fn, operands);
}

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &builder,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    int64_t rank = 2;
    auto oneAttr = builder.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(rank, oneAttr);
    SmallVector<Value> tiledOperands;
    tiledOperands.emplace_back(
        getSlice(builder, op->getLoc(), op->getOperands()[0], offsets, sizes, strides));
    tiledOperands.emplace_back(
        getSlice(builder, op->getLoc(), op->getOperands()[1], offsets, sizes, strides));

    SmallVector<Type, 4> resultTypes;
    resultTypes.push_back(tiledOperands[1].getType());

    Operation *tiledOp =
        mlir::clone(builder, op, resultTypes, tiledOperands);

    return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
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

struct MatMulOpTiling : public TilingInterface::ExternalModel<MatMulOpTiling, MatMulOp> {
  static Value getSlice(OpBuilder &b, Location loc, Value source,
                      ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides) {
    return TypeSwitch<Type, Value>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        return b.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                           strides);
      })
      .Default([&](Type t) { return nullptr; });
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto padOp = cast<MatMulOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        3/*rank*/, utils::IteratorType::parallel);
    iteratorTypes[0] = utils::IteratorType::reduction;
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto tnsrTy = op->getOperand(0).getType().dyn_cast<TensorType>();
    auto tnsrTy1 = op->getOperand(1).getType().dyn_cast<TensorType>();
    auto shape = tnsrTy.getShape();
    auto shape1 = tnsrTy1.getShape();
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    SmallVector<int64_t, 3> ubs;
    ubs.push_back(shape[0]);
    ubs.push_back(shape1[1]);
    ubs.push_back(shape1[0]);
    // Initialize all the ranges to {zero, one, one}. All the `ub`s are
    // overwritten.
    SmallVector<Range> loopRanges(3, {zero, one, one});
    int i =0;
    for (const auto &ub : ubs)
      loopRanges[i++].size = b.getIndexAttr(ub);
    return loopRanges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &builder,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    int64_t rank = 2;
    llvm::errs()<<"sizes.size:"<<sizes.size()<<"\n";
    auto oneAttr = builder.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(rank, oneAttr);
    SmallVector<Value> tiledOperands;
    SmallVector<OpFoldResult, 2> lhsOffset, rhsOffset;
    lhsOffset.push_back(offsets[0]);lhsOffset.push_back(offsets[2]);
    rhsOffset.push_back(offsets[2]);rhsOffset.push_back(offsets[1]);
    SmallVector<OpFoldResult, 2> opSize; opSize.push_back(sizes[0]); opSize.push_back(sizes[0]);
    tiledOperands.emplace_back(
        getSlice(builder, op->getLoc(), op->getOperands()[0], lhsOffset, opSize, strides));
    tiledOperands.emplace_back(
        getSlice(builder, op->getLoc(), op->getOperands()[1], rhsOffset, opSize, strides));

    SmallVector<Type, 4> resultTypes;
    //auto tnsrTy = op->getResult(0).getType().dyn_cast<TensorType>();
    auto resTy = RankedTensorType::get({8,8}, builder.getI64Type());
    resultTypes.push_back(resTy);

    Operation *tiledOp =
        mlir::clone(builder, op, resultTypes, tiledOperands);
    op->getParentOfType<ModuleOp>()->dump();
    return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.push_back(offsets[0]);resultOffsets.push_back(offsets[1]);
    resultSizes.push_back(sizes[0]);resultSizes.push_back(sizes[1]);
    return success();
  }
};

struct MatMulOpRedTiling : public PartialReductionOpInterface::ExternalModel<MatMulOpRedTiling, MatMulOp> {
  FailureOr<SmallVector<Value>> generateInitialTensorForPartialReduction(
      Operation *op, OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
      ArrayRef<int> reductionDims) const {
        auto opTy = op->getResult(0).getType().dyn_cast<TensorType>();
        assert(opTy);
        auto resTy = RankedTensorType::get({8,8}, opTy.getElementType());
        auto zero = b.create<arith::ConstantOp>(op->getLoc(), opTy.getElementType(), b.getI64IntegerAttr(0));
        auto res = b.create<tensor::SplatOp>(op->getLoc(), zero, resTy);
        SmallVector<Value> resVec; resVec.push_back(res);
        return resVec;
      }

  static Value getSlice(OpBuilder &b, Location loc, Value source,
                      ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides) {
    return TypeSwitch<Type, Value>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        return b.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                           strides);
      })
      .Default([&](Type t) { return nullptr; });
  }
  
  FailureOr<TilingResult>
  tileToPartialReduction(Operation *op, OpBuilder &b, Location loc,
                         ValueRange init, ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes,
                         ArrayRef<int> reductionDims) const {
    int64_t rank = 2;
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(rank, oneAttr);
    SmallVector<Value> tiledOperands;
    SmallVector<OpFoldResult> lhsOffset, rhsOffset, size;
    lhsOffset.push_back(offsets[0]); lhsOffset.push_back(offsets[2]);
    rhsOffset.push_back(offsets[2]); rhsOffset.push_back(offsets[1]);
    size.push_back(sizes[0]);size.push_back(sizes[0]);
    tiledOperands.emplace_back(
        getSlice(b, op->getLoc(), op->getOperands()[0], lhsOffset, size, strides));
    tiledOperands.emplace_back(
        getSlice(b, op->getLoc(), op->getOperands()[1], rhsOffset, size, strides));

    SmallVector<Type, 4> resultTypes;
    resultTypes.push_back(tiledOperands[1].getType());

    Operation *tiledOp =
        mlir::clone(b, op, resultTypes, tiledOperands);

    return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};                        
  }

  FailureOr<MergeResult> mergeReductions(Operation *op, OpBuilder &b,
                                         Location loc, ValueRange partialReduce,
                                         ArrayRef<int> reductionDims) const {
    //op->getParentOfType<ModuleOp>()->dump();
    llvm::errs()<<"in here bro\n";
    return failure();
  }
};

}

void mlir::mini::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, MiniDialect *dialect) {
    mini::AddOp::attachInterface<AddOpTiling>(*ctx);
    mini::MatMulOp::attachInterface<MatMulOpRedTiling>(*ctx);
    mini::MatMulOp::attachInterface<MatMulOpTiling>(*ctx);
  });
}
