#ifndef TTL_TYPES
#define TTL_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Dialect/TTL/TTLDialect.td"

class TTLType<string name, string typeMnemonic, list<Trait> traits = []> : 
        TypeDef<TTLDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TTLIntType : TTLType<"Int", "int"> {
  let summary = "TTL integer type";
}
def TTLFloatType : TTLType<"Float", "float"> {
  let summary = "TTL float type";
}
def TTLVoidType : TTLType<"Void", "void"> {
  let summary = "TTL void type";
}

def TTLTensorType : TTLType<"Tensor", "tensor"> {

  let summary = "Multi-dimensional tensor with fixed number of dimensions";

  let parameters = (ins
    "Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "Type":$elementType,
      "ArrayRef<int64_t>":$shape
    ), [{
      return $_get(elementType.getContext(), elementType, shape);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}

def TTLScalarType : AnyTypeOf<[TTLIntType, TTLFloatType]> {}

def TTLValueType : AnyTypeOf<[
                           TTLIntType,
                           TTLFloatType,
                           TTLTensorType],
                           "TTL value type"> {}

def TTLFuncType : AnyTypeOf<[
                            TTLValueType,
                            TTLVoidType],
                            "TTL function type"> {}

#endif // TTL_TYPES
