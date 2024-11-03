#pragma once

#include "TTLToTensor/TTLToTensor.h"
#include "TTLToLinalg/TTLToLinalg.h"
#include "TTLToScalar/TTLToScalar.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace ttl {

#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

} // namespace ttl
} // namespace mlir
