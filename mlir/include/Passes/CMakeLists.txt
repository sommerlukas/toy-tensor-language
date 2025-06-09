set(LLVM_TARGET_DEFINITIONS TTLPasses.td)
mlir_tablegen(TTLPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRTTLPassesIncGen)
