add_mlir_dialect(TTLOps ttl)
add_mlir_doc(TTLDialect TTLDialect TTL/ -gen-dialect-doc)
add_mlir_doc(TTLOps TTLOps TTL/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TTLAttributes.td)
mlir_tablegen(TTLEnums.h.inc -gen-enum-decls)
mlir_tablegen(TTLEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTTLEnumsIncGen)
