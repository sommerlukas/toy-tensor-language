#include "Dialect/TTL/TTLDialect.h"
#include "Dialect/TTL/TTLAttributes.h"
#include "Dialect/TTL/TTLOps.h"
#include "Dialect/TTL/TTLTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ttl;

#include "Dialect/TTL/TTLOpsDialect.cpp.inc"

void TTLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/TTL/TTLOps.cpp.inc"
      >();
  registerTypes();
}
