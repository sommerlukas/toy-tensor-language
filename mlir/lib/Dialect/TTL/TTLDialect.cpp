#include "Dialect/TTL/TTLDialect.h"
#include "Dialect/TTL/TTLAttributes.h"
#include "Dialect/TTL/TTLOps.h"
#include "Dialect/TTL/TTLTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ttl;

#include "Dialect/TTL/TTLOpsDialect.cpp.inc"

namespace {
struct TTLInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    if (!isa<ttl::Yield>(op) && !isa<ttl::Return>(op)) {
      return;
    }

    for (auto retValue : llvm::zip(valuesToRepl, op->getOperands())) {
      std::get<0>(retValue).replaceAllUsesWith(std::get<1>(retValue));
    }
  }
};
} // namespace

void TTLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/TTL/TTLOps.cpp.inc"
      >();
  registerTypes();
  addInterfaces<TTLInlinerInterface>();
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
