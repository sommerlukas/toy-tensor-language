list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/external/antlr4)

add_definitions(-DANTLR4CPP_STATIC)

# using /MD flag for antlr4_runtime (for Visual C++ compilers only)
set(ANTLR4_WITH_STATIC_CRT OFF)

# Specify the version of the antlr4 library needed for this project.
set(ANTLR4_TAG 4.10.1)

# add external build for antlrcpp
include(ExternalAntlr4Cpp)

# set variable pointing to the antlr tool that supports C++
# this is not required if the jar file can be found under PATH environment
set(ANTLR_EXECUTABLE ${CMAKE_SOURCE_DIR}/external/antlr4/antlr-4.10.1-complete.jar)
# add macros to generate ANTLR Cpp code from grammar
find_package(ANTLR REQUIRED)


add_subdirectory(lib)
add_subdirectory(tools)
