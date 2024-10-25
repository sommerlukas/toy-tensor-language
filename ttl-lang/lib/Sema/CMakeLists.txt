add_library(ttl-sema 
            SemaVisitor.cpp
            Sema.cpp
)

target_include_directories(ttl-sema
    PUBLIC
      ${CMAKE_SOURCE_DIR}/ttl-lang/include/Sema
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ttl-sema
  PRIVATE
    ttl-ast
)
