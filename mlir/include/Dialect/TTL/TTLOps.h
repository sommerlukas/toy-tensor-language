#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "TTLAttributes.h"
#include "TTLTypes.h"

namespace mlir {
namespace ttl {
LogicalResult verifyBinOp(Operation *op);
}

namespace OpTrait {
template <typename ConcreteType>
class ElementwiseOrTensorScalarOrScalarBinaryOp
    : public TraitBase<ConcreteType,
                       ElementwiseOrTensorScalarOrScalarBinaryOp> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return ttl::verifyBinOp(op);
  }
};
} // namespace OpTrait
} // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/TTL/TTLOps.h.inc"
