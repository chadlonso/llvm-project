//===-- Passes.td - Arith pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MINI_TRANSFORMS_PASSES
#define MLIR_DIALECT_MINI_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MiniTiling : Pass<"tile-mini-ops", "func::FuncOp"> {
  let summary = "Unroll and jam affine loops";
  let constructor = "mlir::mini::createTilingPass()";
  let dependentDialects = [
      "affine::AffineDialect",
      "tensor::TensorDialect", "scf::SCFDialect"
  ];
}


#endif // MLIR_DIALECT_MINI_TRANSFORMS_PASSES
