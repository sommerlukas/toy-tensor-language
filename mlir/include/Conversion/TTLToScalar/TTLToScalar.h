#pragma once

#include <memory>

namespace mlir {
class Pass;
namespace ttl {

#define GEN_PASS_DECL_TTLTOSCALAR
#include "Conversion/Passes.h.inc"
} // namespace ttl
} // namespace mlir
