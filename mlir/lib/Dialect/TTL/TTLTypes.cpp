#include "Dialect/TTL/TTLTypes.h"

#include "Dialect/TTL/TTLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ttl;

LogicalResult parseTTLShape(AsmParser &parser,
                            llvm::SmallVector<int64_t> &shape) {
  return parser.parseDimensionList(shape, true, false);
}
void printTTLShape(AsmPrinter &printer, llvm::ArrayRef<int64_t> shape) {
  printer.printDimensionList(shape);
}

#define GET_TYPEDEF_CLASSES
#include "Dialect/TTL/TTLOpsTypes.cpp.inc"

void TTLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/TTL/TTLOpsTypes.cpp.inc"
      >();
}
