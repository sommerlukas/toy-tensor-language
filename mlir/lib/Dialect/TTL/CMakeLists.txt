add_mlir_dialect_library(MLIRTTL
        TTLDialect.cpp
        TTLOps.cpp
        TTLAttributes.cpp
        TTLTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/mlir/include/Dialect/TTL

        DEPENDS
        MLIRTTLOpsIncGen
        MLIRTTLEnumsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
	)
