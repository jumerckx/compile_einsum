"func.func"() <{function_type = (memref<?xf64, strided<[1]>>, memref<?x?xf64, strided<[1, ?]>>, memref<?xf64, strided<[1]>>) -> i64, sym_name = "scaffolding"}> ({
^bb0(%arg0: memref<?xf64, strided<[1]>>, %arg1: memref<?x?xf64, strided<[1, ?]>>, %arg2: memref<?xf64, strided<[1]>>):
  %0 = "arith.constant"() <{value = 1 : i64}> : () -> i64
  %1 = "arith.constant"() <{value = 1 : index}> : () -> index
  %2 = "index.casts"(%0) : (i64) -> index
  %3 = "index.sub"(%2, %1) : (index, index) -> index
  %4 = "memref.dim"(%arg0, %3) : (memref<?xf64, strided<[1]>>, index) -> index
  %5 = "index.casts"(%4) : (index) -> i64
  %6 = "arith.constant"() <{value = 2 : i64}> : () -> i64
  %7 = "arith.constant"() <{value = 1 : index}> : () -> index
  %8 = "index.casts"(%6) : (i64) -> index
  %9 = "index.sub"(%8, %7) : (index, index) -> index
  %10 = "memref.dim"(%arg1, %9) : (memref<?x?xf64, strided<[1, ?]>>, index) -> index
  %11 = "index.casts"(%10) : (index) -> i64
  %12 = "arith.addi"(%5, %11) : (i64, i64) -> i64
  "func.return"(%12) : (i64) -> ()
}) {llvm.emit_c_interface} : () -> ()
