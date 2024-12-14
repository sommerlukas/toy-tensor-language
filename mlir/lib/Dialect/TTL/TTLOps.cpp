#include "Dialect/TTL/TTLDialect.h"
#include "Dialect/TTL/TTLOps.h"
#include "Dialect/TTL/TTLTypes.h"
#include "Dialect/TTL/TTLAttributes.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Dialect/TTL/TTLOps.cpp.inc"

using namespace mlir;
using namespace mlir::ttl;

void ForLoop::build(OpBuilder &builder, OperationState &result, Value lowerBound, Value upperBound, Value step, ValueRange initArgs) {
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands({lowerBound, upperBound, step});
  result.addOperands(initArgs);


  Type lowerBoundTy = lowerBound.getType();
  
  // Initialize the for-loop with a region and the single block for the loop body.
  Region* bodyRegion = result.addRegion();
  Block* bodyBlock = builder.createBlock(bodyRegion);
  bodyBlock->addArgument(lowerBoundTy, result.location);
  for(Value v : initArgs) {
    result.addTypes(v.getType());
    bodyBlock->addArgument(v.getType(), v.getLoc());
  }
}
