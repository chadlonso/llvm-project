//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MINI_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_MINI_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class DataFlowSolver;
class ConversionTarget;
class TypeConverter;

namespace func {
class FuncOp;
} // namespace func
class AffineDialect;
namespace tensor {
class TensorDialect;
} // namespace tensor
namespace scf {
class SCFDialect;
} // namespace scf
namespace mini {

#define GEN_PASS_DECL
#include "mlir/Dialect/Mini/Transforms/Passes.h.inc"

std::unique_ptr<OperationPass<func::FuncOp>>
createTilingPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Mini/Transforms/Passes.h.inc"

} // namespace mini
} // namespace mlir

#endif // MLIR_DIALECT_MINI_TRANSFORMS_PASSES_H_
