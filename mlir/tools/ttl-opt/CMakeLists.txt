get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
message(STATUS "${dialect_libs}")
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        )
add_llvm_executable(ttl-opt ttl-opt.cpp)

llvm_update_compile_flags(ttl-opt)
target_link_libraries(ttl-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(ttl-opt)
