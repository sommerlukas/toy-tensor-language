#ifndef TTL_DIALECT
#define TTL_DIALECT

include "mlir/IR/OpBase.td"

def TTLDialect : Dialect {
  let name = "ttl";
  let summary = "A dialect for the Toy Tensor Language (TTL)";
  let cppNamespace = "::mlir::ttl";

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
        void registerTypes();
  }];

  let hasConstantMaterializer = 1;
}

class TTLOp<string mnemonic, list<Trait> traits = []> : 
        Op<TTLDialect, mnemonic, traits>;


#endif // TTL_DIALECT
