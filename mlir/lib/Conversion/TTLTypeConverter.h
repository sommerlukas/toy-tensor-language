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
    addConversion([&](ttl::FloatType type) {
      return Float32Type::get(type.getContext());
    });
    addConversion([&](ttl::VoidType type) {
      return mlir::NoneType::get(type.getContext());
    });
    addConversion([&](ttl::TensorType type) {
      return RankedTensorType::get(type.getShape(),
                                   convertType(type.getElementType()));
    });

    addArgumentMaterialization([&](OpBuilder &builder, Type resultType,
                                   ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  };
};

} // namespace ttl
} // namespace mlir
