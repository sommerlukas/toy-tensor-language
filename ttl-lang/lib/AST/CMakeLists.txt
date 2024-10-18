add_library(ttl-ast 
            ASTContext.cpp
            ASTBuilder.cpp
)

target_include_directories(ttl-ast
    PUBLIC
      ${CMAKE_SOURCE_DIR}/ttl-lang/include/AST
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ttl-ast
  PRIVATE
    ttl-parser
    antlr4_static
)
