add_mlir_library(TTLPasses
  EliminateInitLoops.cpp

  DEPENDS
  MLIRTTLPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRPass
  )
