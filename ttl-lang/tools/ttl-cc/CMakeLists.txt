add_executable(ttl-cc ttl-cc.cpp)

llvm_map_components_to_libnames(llvm_libs support core)

target_link_libraries(ttl-cc 
                      ttl-parser 
                      ttl-ast 
                      ttl-sema
                      ttl-codegen
                      ${llvm_libs}
                      MLIRIR
                      MLIRTTL
                      MLIRFuncDialect
                      MLIRSCFDialect
)

