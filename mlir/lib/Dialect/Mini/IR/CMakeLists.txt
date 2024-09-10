set(LLVM_OPTIONAL_SOURCES
  MiniOps.cpp
  MiniDialect.cpp
#  InferIntRangeInterfaceImpls.cpp
#  ValueBoundsOpInterfaceImpl.cpp
  )

add_mlir_dialect_library(MLIRMiniDialect
  MiniOps.cpp
  MiniDialect.cpp
  #InferIntRangeInterfaceImpls.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Mini

  DEPENDS
  MLIRMiniOpsIncGen

  LINK_LIBS PUBLIC
  MLIRCastInterfaces
  MLIRDialect
  MLIRInferIntRangeCommon
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRUBDialect
  )
