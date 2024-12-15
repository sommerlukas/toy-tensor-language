#include "Dialect/TTL/TTLDialect.h"
#include "Dialect/TTL/TTLAttributes.h"
#include "Dialect/TTL/TTLOps.h"
#include "Dialect/TTL/TTLTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
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

Operation *TTLDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type ty, Location loc) {
  if (isa<ttl::FloatType>(ty) && isa<FloatAttr>(value)) {
    return builder.create<ttl::FloatConstant>(loc, ty, cast<FloatAttr>(value));
  }
  if (isa<ttl::IntType>(ty) && isa<IntegerAttr>(value)) {
    return builder.create<ttl::IntConstant>(loc, ty, cast<IntegerAttr>(value));
  }
  return nullptr;
}
