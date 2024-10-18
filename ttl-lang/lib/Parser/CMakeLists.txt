# add antrl4cpp artifacts to project environment
include_directories(${ANTLR4_INCLUDE_DIRS})

# Call macro to add lexer and grammar to your build dependencies.
antlr_target(TTLParser ${CMAKE_CURRENT_SOURCE_DIR}/TTL.g4
             PACKAGE "ttl::parser"
             VISITOR)

add_library(ttl-parser
              TTLANTLRParser.cpp
              ${ANTLR_TTLParser_CXX_OUTPUTS}
)

target_include_directories(ttl-parser
    PUBLIC
      ${CMAKE_SOURCE_DIR}/ttl-lang/include/Parser
      ${ANTLR4_INCLUDE_DIRS}
      ${ANTLR_TTLParser_OUTPUT_DIR}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ttl-parser
    PUBLIC
      antlr4_static
)
