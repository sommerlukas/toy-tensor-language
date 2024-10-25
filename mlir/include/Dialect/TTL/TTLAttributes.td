include "Dialect/TTL/TTLDialect.td"
include "mlir/IR/EnumAttr.td"

def TTLCompareOpcodes : I32EnumAttr<"TTLCmpOpcodes", "TTL compare opcodes", [
                                    I32EnumAttrCase<"GT", 0, "gt">,
                                    I32EnumAttrCase<"LT", 1, "lt">,
                                    I32EnumAttrCase<"GE", 2, "ge">,
                                    I32EnumAttrCase<"LE", 3, "le">,
                                    I32EnumAttrCase<"EQ", 4, "eq">,
                                    I32EnumAttrCase<"NE", 5, "ne">]> {
  let cppNamespace = "::mlir::ttl";

}

def TTLCompareOpcodesAttr : EnumAttr<TTLDialect, TTLCompareOpcodes, "compare">;

