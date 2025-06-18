#pragma once

#include "Dialect/TTL/TTLDialect.h"
#include "Dialect/TTL/TTLOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace ttl {
#define GEN_PASS_DECL
#include "Transform/TTLPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Transform/TTLPasses.h.inc"

} // namespace ttl
} // namespace mlir
