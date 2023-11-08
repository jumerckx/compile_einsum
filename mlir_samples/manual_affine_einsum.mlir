func.func @scaffolding(%arg0: memref<?xf64, strided<[1]>>, %arg1: memref<?x?xf64, strided<[1, ?]>>, %arg2: memref<?xf64, strided<[1]>>) -> i64 attributes {llvm.emit_c_interface} {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %dim = memref.dim %arg0, %idx0 : memref<?xf64, strided<[1]>>
  %0 = index.casts %dim : index to i64
  %dim_0 = memref.dim %arg1, %idx1 : memref<?x?xf64, strided<[1, ?]>>
  %1 = index.casts %dim_0 : index to i64

  affine.for %i = 0 to %dim {
    %accum_0 = arith.constant 0.0: f64

    %accum = affine.for %j = 0 to %dim_0
        iter_args(%accum_iter = %accum_0) -> f64 {
      
      // %b = "memref.load"(%arg1, %i, %j) : (memref<?x?xf64, strided<[1, ?]>>, index, index) -> f64
      %b = affine.load %arg1[%i, %j] : memref<?x?xf64, strided<[1, ?]>>

      // %c = "memref.load"(%arg1, %j) : (memref<?xf64, strided<[1]>>, index) -> f64
      %c = affine.load %arg2[%j] : memref<?xf64, strided<[1]>>
      %s = arith.mulf %b, %c : f64
      %accum_next = arith.addf %s, %accum_iter : f64
      affine.yield %accum_next : f64
    }

    affine.store %accum, %arg0[%i] : memref<?xf64, strided<[1]>>
  }

  %2 = arith.addi %0, %1 : i64
  return %2 : i64
}
