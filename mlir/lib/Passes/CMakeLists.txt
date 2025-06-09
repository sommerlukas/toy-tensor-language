add_mlir_library(TTLPasses
  TTLTiling.cpp

  DEPENDS
  MLIRTTLPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRPass
  MLIRTilingInterface
  )
