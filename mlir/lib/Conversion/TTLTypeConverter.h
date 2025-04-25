#include "Dialect/TTL/TTLTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace ttl {

class TTLTypeConverter : public TypeConverter {

public:
  TTLTypeConverter() {

    addConversion([&](ttl::IntType type) {
      return IntegerType::get(type.getContext(), 32);
    });
    addConversion([&](mlir::IntegerType type) { return type; });
    addConversion([&](ttl::FloatType type) {
      return Float32Type::get(type.getContext());
    });
    addConversion([&](mlir::FloatType type) { return type; });
    addConversion([&](ttl::VoidType type) {
      return mlir::NoneType::get(type.getContext());
    });
    addConversion([&](mlir::NoneType type) { return type; });
    addConversion([&](ttl::TensorType type) {
      return RankedTensorType::get(type.getShape(),
                                   convertType(type.getElementType()));
    });
    addConversion([&](mlir::RankedTensorType type) { return type; });

    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  };
};

} // namespace ttl
} // namespace mlir
