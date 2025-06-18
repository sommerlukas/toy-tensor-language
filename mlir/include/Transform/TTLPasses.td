#ifndef TTL_PASSES
#define TTL_PASSES

include "mlir/Pass/PassBase.td"

def TTLEliminateInitLoops : Pass<"ttl-eliminate-init-loops", 
                                  "::mlir::func::FuncOp"> {
  let summary = "Replaces loops initializing empty tensors with a range init";

  let statistics = [
    Statistic<"numLoopsEliminated", "num-loops-eliminated", 
                "Number of init loops eliminated">
  ];
}

#endif // TTL_PASSES
