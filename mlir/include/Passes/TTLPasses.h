#pragma once

#include "Dialect/TTL/TTLDialect.h"
#include "Dialect/TTL/TTLOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace ttl {

#define GEN_PASS_DECL
#include "Passes/TTLPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Passes/TTLPasses.h.inc"
} // namespace ttl
} // namespace mlir
