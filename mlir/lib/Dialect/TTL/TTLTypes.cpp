#include "Dialect/TTL/TTLTypes.h"

#include "Dialect/TTL/TTLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ttl;

mlir::Type ttl::TensorType::parse(::mlir::AsmParser &parser) {
  if (parser.parseLess())
    return {};

  llvm::SmallVector<int64_t> shape;

  auto shapeLoc = parser.getCurrentLocation();
  if (mlir::failed(parser.parseDimensionList(shape))) {
    parser.emitError(shapeLoc, "failed to parse parameter 'shape'");
    return {};
  }

  mlir::Type elementType;
  auto elemTypeLoc = parser.getCurrentLocation();
  if (mlir::failed(parser.parseType(elementType))) {
    parser.emitError(elemTypeLoc, "failed to parse parameter 'elementType'");
    return {};
  }

  if (parser.parseGreater())
    return {};

  return ttl::TensorType::get(parser.getContext(), elementType, shape);
}

void ttl::TensorType::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  printer << "x";
  printer << getElementType();
  printer << ">";
}

#define GET_TYPEDEF_CLASSES
#include "Dialect/TTL/TTLOpsTypes.cpp.inc"

void TTLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/TTL/TTLOpsTypes.cpp.inc"
      >();
}
