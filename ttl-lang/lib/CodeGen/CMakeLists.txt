add_library(ttl-codegen 
            CodeGen.cpp
            CodeGenVisitor.cpp
)

target_include_directories(ttl-codegen
    PUBLIC
      ${CMAKE_SOURCE_DIR}/ttl-lang/include/CodeGen
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ttl-codegen
  PRIVATE
    ttl-ast
    MLIRIR
    MLIRTTL
    MLIRFuncDialect
    MLIRSCFDialect
)
