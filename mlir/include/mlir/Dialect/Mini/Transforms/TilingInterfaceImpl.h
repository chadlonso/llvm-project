//===- MiniTilingOpInterfaceImpl.h - ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Tiling interface for MiniOps with ExternalModel.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MINI_TILINGINTERFACEIMPL_H_
#define MLIR_DIALECT_MINI_TILINGINTERFACEIMPL_H_

#include "mlir/IR/Dialect.h"

namespace mlir {

class DialectRegistry;

namespace mini {
/// Registers external models for Tiling interface for mini ops.
/// Currently, it registers:
///
/// * TilingInterface for `mini.pad`, `mini.pack`, and `mini.unpack`.
///
/// Unfortunately, a "normal" internal registration is not possible at the
/// moment, because of the dependency of the interface implementation for these
/// ops on `affine.apply` and Affine dialect already depends on MiniOps. In
/// order to break the cyclic dependency (MiniOps->AffineOps->MiniOps) the
/// implementation is moved to a separate library.
void registerTilingInterfaceExternalModels(mlir::DialectRegistry &registry);
} // namespace mini
} // namespace mlir

#endif // MLIR_DIALECT_MINI_IR_MINITILINGINTERFACEIMPL_H_
