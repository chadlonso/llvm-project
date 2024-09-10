//===- MiniDialect.cpp - MLIR Mini dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Mini/IR/Mini.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::mini;

#include "mlir/Dialect/Mini/IR/MiniOpsDialect.cpp.inc"
// #include "mlir/Dialect/Mini/IR/MiniOpsInterfaces.cpp.inc"
// #define GET_ATTRDEF_CLASSES
// #include "mlir/Dialect/Mini/IR/MiniOpsAttributes.cpp.inc"

// namespace {
// /// This class defines the interface for handling inlining for minimetic
// /// dialect operations.
// struct MiniInlinerInterface : public DialectInlinerInterface {
//   using DialectInlinerInterface::DialectInlinerInterface;

//   /// All minimetic dialect ops can be inlined.
//   bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
//     return true;
//   }
// };
// } // namespace

void mini::MiniDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Mini/IR/MiniOps.cpp.inc"
      >();
  declarePromisedInterfaces<bufferization::BufferizableOpInterface,
#define GET_OP_LIST
#include "mlir/Dialect/Mini/IR/MiniOps.cpp.inc"
                            >();
  declarePromisedInterfaces<TilingInterface,
#define GET_OP_LIST
#include "mlir/Dialect/Mini/IR/MiniOps.cpp.inc"
                            >();
//   addAttributes<
// #define GET_ATTRDEF_LIST
// #include "mlir/Dialect/Mini/IR/MiniOpsAttributes.cpp.inc"
//       >();
//   addInterfaces<MiniInlinerInterface>();
//   declarePromisedInterface<ConvertToLLVMPatternInterface, MiniDialect>();
//   declarePromisedInterface<bufferization::BufferDeallocationOpInterface,
//                            SelectOp>();
//   declarePromisedInterfaces<bufferization::BufferizableOpInterface, ConstantOp,
//                             IndexCastOp, SelectOp>();
//   declarePromisedInterfaces<ValueBoundsOpInterface, AddIOp, ConstantOp, SubIOp,
//                             MulIOp>();
}
