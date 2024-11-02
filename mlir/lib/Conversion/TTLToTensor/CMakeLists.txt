add_mlir_conversion_library(TTLToTensor
  TTLToTensor.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/Conversion

  DEPENDS
  MLIRTTLConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  )
