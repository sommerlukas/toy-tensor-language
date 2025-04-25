# Toy Tensor Language

As the name indicates, the Toy Tensor Language (TTL) is not intended to be a
useful programming language. Instead, it is intended as a vehicle to demonstrate
MLIR-based compiler engineering with an end-to-end compiler flow using MLIR.

As such, many elements of the language are designed to map easily to MLIR
constructs and dialects.

The full language documentation can be found [here](doc/language.md).

The Toy Tensor Language Compiler (`ttlc`) is fully functional compiler that can
compile from source input to executable.

## Setup

There are currently no binary packages available for the compiler, so you will
have to build it from source, following these instructions.

First, choose a directory where you want the build to live and export it as
`BASE_DIR`, e.g.:

```sh
export BASE_DIR=/tmp/ttlc
```

Change directory to your base directory: `cd $BASE_DIR`. 

First, clone this repository:

```sh
git clone git@github.com:sommerlukas/toy-tensor-language.git
```

Next, clone and build LLVM, using the known-good base commit specified in this
repository:

```sh
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout $(cat $BASE_DIR/toy-tensor-language/external/llvm/llvm-hash.txt)
mkdir build
cd build
cmake -G Ninja ../llvm \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_ENABLE_PROJECTS="mlir;clang" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
ninja
```

`ttlc` uses an ANTLR4 parser, so download the 4.10.1 release runtime of ANTLR4.

```sh
wget -P $BASE_DIR/toy-tensor-language/external/antlr4 \
  https://github.com/antlr/website-antlr4/raw/refs/heads/gh-pages/download/antlr-4.10.1-complete.jar
```

Lastly, build the Toy Tensor Language Compiler itself:

```sh
cd $BASE_DIR/toy-tensor-language
mkdir build
cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_PREFIX_PATH=$BASE_DIR/llvm-project/build/lib/cmake/ \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    ..
ninja
```

If you want to test whether the build was successful, run the tests:

```sh
ninja check-ttl
```
