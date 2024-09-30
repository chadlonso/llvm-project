//===- MiniOps.td - Mini op definitions ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MINI_OPS
#define MINI_OPS

include "mlir/Dialect/Mini/IR/MiniBase.td"
//include "mlir/Dialect/Mini/IR/MiniOpsInterfaces.td"
//include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
//include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
//include "mlir/IR/EnumAttr.td"

// Base class for Mini dialect ops. Ops in this dialect have no memory
// effects and can be applied element-wise to vectors and tensors.
class Mini_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mini_Dialect, mnemonic,
       traits #
       [NoMemoryEffect]>;

// Base class for integer and floating point minimetic ops. All ops have one
// result, require operands and results to be of the same type, and can accept
// tensors or vectors of integers or floats.
class Mini_MiniOp<string mnemonic, list<Trait> traits = []> :
    Mini_Op<mnemonic, traits # []>;

// Base class for binary minimetic operations.
class Mini_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Mini_MiniOp<mnemonic, traits> {
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

// Base class for integer binary operations.
class Mini_IntBinaryOp<string mnemonic, list<Trait> traits = []> :
    Mini_BinaryOp<mnemonic, traits #
      []>,
    Arguments<(ins AnyType:$lhs, AnyType:$rhs)>,
    Results<(outs AnyType:$result)>;

// Base class for integer binary operations without undefined behavior.
class Mini_TotalIntBinaryOp<string mnemonic, list<Trait> traits = []> :
    Mini_IntBinaryOp<mnemonic, traits # [Pure]>;

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

def Mini_AddOp : Mini_TotalIntBinaryOp<"add", [Commutative
      //,DeclareOpInterfaceMethods<TilingInterface,
      //["getIterationDomain",
       //"getLoopIteratorTypes",
       //"getResultTilePosition",
       //"getTiledImplementation"]>
       ]> {
  let summary = "integer addition operation";
  let description = [{
    The `addi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type, 
    a vector whose element type is integer, or a tensor of integers.
  }];
  //let hasFolder = 1;
  //let hasCanonicalizer = 1;
}

def Mini_MatMulOp : Mini_TotalIntBinaryOp<"matmul", [
  //Commutative
      //,DeclareOpInterfaceMethods<TilingInterface,
      //["getIterationDomain",
       //"getLoopIteratorTypes",
       //"getResultTilePosition",
       //"getTiledImplementation"]>
       ]> {
  let summary = "integer addition operation";
  let description = [{
    The `addi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type, 
    a vector whose element type is integer, or a tensor of integers.
  }];
  //let hasFolder = 1;
  //let hasCanonicalizer = 1;
}

#endif // MINI_OPS
