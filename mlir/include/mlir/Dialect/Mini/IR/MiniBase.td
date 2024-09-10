//===- MiniBase.td - Base defs for mini dialect -----------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MINI_BASE
#define MINI_BASE

include "mlir/IR/OpBase.td"

def Mini_Dialect : Dialect {
  let name = "mini";
  let cppNamespace = "::mlir::mini";
  let description = [{
    The mini dialect is intended to hold basic integer and floating point
    mathematical operations. This includes unary, binary, and ternary minimetic
    ops, bitwise and shift ops, cast ops, and compare ops. Operations in this
    dialect also accept vectors and tensors of integers or floats. The dialect
    assumes integers are represented by bitvectors with a two's complement 
    representation. Unless otherwise stated, the operations within this dialect 
    propagate poison values, i.e., if any of its inputs are poison, then the 
    output is poison. Unless otherwise stated, operations applied to `vector` 
    and `tensor` values propagates poison elementwise.
  }];

  //let hasConstantMaterializer = 1;
  //let useDefaultAttributePrinterParser = 1;
}

#endif // MINI_BASE
