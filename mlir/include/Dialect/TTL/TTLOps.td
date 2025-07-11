#ifndef MLIR_INCLUDE_DIALECT_TTL_TTLOPS
#define MLIR_INCLUDE_DIALECT_TTL_TTLOPS

include "Dialect/TTL/TTLDialect.td"
include "Dialect/TTL/TTLAttributes.td"
include "Dialect/TTL/TTLTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"

class TTLArithBinaryOp<string mnemonic> :
        TTLOp<mnemonic, [ElementwiseOrTensorScalarOrScalarBinaryOp]> {
  let summary = "TTL arithmetic binary operation";

  let arguments = (ins TTLValueType:$left, TTLValueType:$right);

  let results = (outs TTLValueType:$opResult);
}

class TTLLogicalBinaryOp<string mnemonic> : TTLOp<mnemonic, [Pure]> {
  let summary = "TTL logical binary operation";

  let arguments = (ins TTLIntType:$left, TTLIntType:$right);

  let results = (outs TTLIntType:$opResult);
}

class TTLUnaryOp<string mnemonic> :
        TTLOp<mnemonic, [SameOperandsAndResultType]> {
  let summary = "TTL unary operation";

  let arguments = (ins TTLValueType:$op);

  let results = (outs TTLValueType:$opResult);
}

def Add : TTLArithBinaryOp<"add">;
def Sub : TTLArithBinaryOp<"sub">;
def Mul : TTLArithBinaryOp<"mul">;
def Div : TTLArithBinaryOp<"div">;
def And : TTLLogicalBinaryOp<"and">;
def Or : TTLLogicalBinaryOp<"or">;

def MatMul : TTLOp<"matmul"> {
  let summary = "TTL matrix multiplication operation";
  
  let arguments = (ins TTLTensorType:$left, TTLTensorType:$right);

  let results = (outs TTLTensorType:$opResult);

  let hasVerifier = 1;
}

def Minus : TTLUnaryOp<"minus">;
def Not : TTLUnaryOp<"not">;

def Compare : TTLOp<"cmp"> {
  let summary = "TTL comparison operation";

  let arguments = (ins TTLScalarType:$left, TTLScalarType:$right, 
                      TTLCompareOpcodesAttr:$opcode);

  let results = (outs TTLIntType:$cmpResult);
}

def TensorEmpty : TTLOp<"tensor_empty"> {
  let summary = "TTL initialize empty tensor";

  let results = (outs TTLTensorType:$tensor);
}

def TensorScalarInit : TTLOp<"tensor_scalar_init"> {
  let summary = "TTL initialize tensor with scalar value";

  let arguments = (ins TTLScalarType:$initVal);

  let results = (outs TTLTensorType:$tensor);
}

def TensorRangeInit : TTLOp<"tensor_range_init"> {
  let summary = "TTL initialize tensor with range";

  let arguments = (ins TTLIntType:$start, TTLIntType:$end);

  let results = (outs TTLTensorType:$tensor);
}

def TensorListInit : TTLOp<"tensor_list_init"> {
  let summary = "TTL initialize tensor";

  let arguments = (ins Variadic<TTLScalarType>:$elements);

  let results = (outs TTLTensorType:$tensor);

  let hasCanonicalizer = 1;
}

def TensorInsert : TTLOp<"tensor_insert"> {
  let summary = "TTL assign tensor element";

  let arguments = (ins TTLTensorType:$dest, TTLScalarType:$value, 
                        Variadic<TTLIntType>:$indices);

  let results = (outs TTLTensorType:$tensor);
}

def Dim : TTLOp<"dim"> {
  let summary = "TTL dimension size query";

  let arguments = (ins TTLTensorType:$tensor, TTLIntType:$dimension);

  let results = (outs TTLIntType:$dimSize);

  let hasFolder = 1;
}

def Slice : TTLOp<"slice", [SameVariadicOperandSize]> {
  let summary = "TTL slice of tensor";

  let arguments = (ins TTLTensorType:$tensor, Variadic<TTLIntType>:$offsets, 
                        Variadic<TTLIntType>:$sizes);

  let results = (outs TTLValueType:$slice);

  let hasFolder = 1;
}

def Yield : TTLOp<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = "TTL control flow termination";

  let arguments = (ins Variadic<TTLValueType>:$results);
}

def Return : TTLOp<"return", [ReturnLike, Terminator,
                              HasParent<"func::FuncOp">]> {
  let summary = "TTL function return";

  let arguments = (ins TTLValueType:$retVal);

  let hasVerifier = 1;
}

def IntConstant : TTLOp<"const_int", [Pure, ConstantLike]> {
  let summary = "TTL constant integer";

  let arguments = (ins I32Attr:$constVal);

  let results = (outs TTLIntType:$val);

  let hasFolder = 1;

  let assemblyFormat = [{ attr-dict $constVal }];
}

def FloatConstant : TTLOp<"const_float", [Pure, ConstantLike]> {
  let summary = "TTL constant float";

  let arguments = (ins F32Attr:$constVal);

  let results = (outs TTLFloatType:$val);

  let hasFolder = 1;
}

def If : TTLOp<"if"> {
  let summary = "TTL if statement";

  let arguments = (ins TTLIntType:$cond);

  let results = (outs Variadic<TTLValueType>:$results);

  let regions = (region SizedRegion<1>:$thenRegion,
                        MaxSizedRegion<1>:$elseRegion);
  
  let assemblyFormat = [{
    $cond `:` functional-type(operands, results) $thenRegion
      (`else` $elseRegion^)? attr-dict
  }];
}

def ForLoop : TTLOp<"for", [SingleBlockImplicitTerminator<"ttl::Yield">, 
    DeclareOpInterfaceMethods<LoopLikeOpInterface,
       ["getInitsMutable", "getLoopResults", "getRegionIterArgs",
        "getLoopInductionVars", "getLoopLowerBounds", "getLoopSteps",
        "getLoopUpperBounds", "getYieldedValuesMutable"]>]> {
  let summary = "TTL for loop";

  let arguments = (ins TTLIntType:$lowerBound, 
                        TTLIntType:$upperBound, 
                        TTLIntType:$step, 
                        Variadic<TTLValueType>:$initArgs);

  let results = (outs Variadic<TTLValueType>:$results);

  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$lowerBound, "Value":$upperBound, "Value":$step, 
                    CArg<"ValueRange", "std::nullopt">:$initArgs)>
  ];

  let extraClassDeclaration = [{
    
    Value getInductionVar() { return getBody()->getArgument(0); }

  }];
}


// TODO: Compare, Constant
// TODO: For, If, Return

#endif // MLIR_INCLUDE_DIALECT_TTL_TTLOPS
