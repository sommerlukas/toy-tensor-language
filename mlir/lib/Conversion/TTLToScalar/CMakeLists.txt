add_mlir_conversion_library(TTLToScalar
  TTLToScalar.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/Conversion
  ${PROJECT_SOURCE_DIR}/mlir/lib/Conversion

  DEPENDS
  MLIRTTLConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRPass
  MLIRSCFDialect
  MLIRArithDialect
  )
