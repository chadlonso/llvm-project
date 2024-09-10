set(LLVM_TARGET_DEFINITIONS MiniOps.td)
#mlir_tablegen(MiniOpsEnums.h.inc -gen-enum-decls)
#mlir_tablegen(MiniOpsEnums.cpp.inc -gen-enum-defs)
#mlir_tablegen(MiniOpsAttributes.h.inc -gen-attrdef-decls
#              -attrdefs-dialect=mini)
#mlir_tablegen(MiniOpsAttributes.cpp.inc -gen-attrdef-defs
#              -attrdefs-dialect=mini)
add_mlir_dialect(MiniOps mini)
add_mlir_doc(MiniOps MiniOps Dialects/ -gen-dialect-doc)
#add_mlir_interface(MiniOpsInterfaces)
