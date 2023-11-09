using MLIR, MLIR_jll
includet("utils.jl")
includet("Brutus.jl")

using Einsum, BenchmarkTools, MLIR, MacroTools

using MLIR.Dialects.Affine
using MLIR.IR


using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

IR.Attribute(API.mlirAffineMapAttrGet(API.mlirAffineMapConstantGet(context().context, 10)))

IR.Attribute(API.mlirStridedLayoutAttrGet(IR.context().context, 0, 2, Int[1, typemin(Int)]))

IR.MLIRType(API.mlirMemRefTypeGet(
    IR.MLIRType(Float64),
    3,
    Int[typemin(Int), typemin(Int), typemin(Int)],
    IR.Attribute(API.mlirStridedLayoutAttrGet(IR.context().context, 0, 3, Int[1, typemin(Int), typemin(Int)])),
    IR.Attribute()))

mod = parse(IR.MModule, """
func.func @should_fuse_reduction_to_pointwise() {
  %a = memref.alloc() : memref<10x10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>

  %cf7 = arith.constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %v0 = affine.load %b[%i0] : memref<10xf32>
      %v1 = affine.load %a[%i0, %i1] : memref<10x10xf32>
      %v3 = arith.addf %v0, %v1 : f32
      affine.store %v3, %b[%i0] : memref<10xf32>
    }
  }
  affine.for %i2 = 0 to 10 {
    %v4 = affine.load %b[%i2] : memref<10xf32>
    affine.store %v4, %c[%i2] : memref<10xf32>
  }
  return
}


""")

op = IR.get_operation(mod)

temp = []

for x in IR.OperationIterator(first(IR.BlockIterator(first(IR.RegionIterator(first(IR.OperationIterator(first(IR.BlockIterator(first(IR.RegionIterator(op)))))))))))
    push!(temp, x)
end

affine_for = temp[5]


B = rand(1000, 2000);
C = rand(2000);
A = similar(B, (1000, ));

function scaffolding(a::AbstractVector{T}, b::AbstractMatrix{T}, c::AbstractVector{T}) where T
  i_max = size(a, 1)
  j_max = size(b, 2)

  for i in 1:i_max
    temp = eltype(a)(0)
    for j in 1:j_max
      temp += b[i, j] * c[j]
    end
    a[i] = temp
  end
  a[i_max] = b[i_max, j_max]*c[j_max]
  return 1
end

@time scaffolding(A, B, C)

op = Brutus.@code_mlir scaffolding(A, B, C)
mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)
pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)

addr = jit(mod; opt=3)("_mlir_ciface_scaffolding")

@ccall $addr(MemRef(A)::Ref{MemRef}, MemRef(B)::Ref{MemRef}, MemRef(C)::Ref{MemRef})::Int


prettify(@expand @einsum A[i] = B[i, j] * C[j])

prettify(@expand @einsum A[i, j] = B[i, k] * C[k, j])

f(a, b, c) = @einsum a[i] = b[i, j] * c[j];

@btime f($A, $B, $C)

mod = parse(IR.MModule, """
#map = affine_map<(i) -> (i-1)>
func.func @scaffolding(%arg0: memref<?xf64, strided<[1]>>, %arg1: memref<?x?xf64, strided<[1, ?]>>, %arg2: memref<?xf64, strided<[1]>>) -> i64 attributes {llvm.emit_c_interface} {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %dim = memref.dim %arg0, %idx0 : memref<?xf64, strided<[1]>>
  %0 = index.casts %dim : index to i64
  %dim_0 = memref.dim %arg1, %idx1 : memref<?x?xf64, strided<[1, ?]>>
  %1 = index.casts %dim_0 : index to i64

  affine.for %i = 0 to %dim {
    %accum_0 = arith.constant 0.0: f64

    %i_ = affine.apply #map(%i)

    %accum = affine.for %j = 0 to %dim_0
        iter_args(%accum_iter = %accum_0) -> f64 {
      
      %j_ = affine.apply #map(%j)
      
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
""")
pm = lowerModuleToLLVM(mod)

op = IR.get_operation(mod)

addr = jit(mod; opt=3)("_mlir_ciface_scaffolding")

@ccall $addr(MemRef(A)::Ref{MemRef}, MemRef(B)::Ref{MemRef}, MemRef(C)::Ref{MemRef})::Int

@btime ccall(addr, Int, (Ref{MemRef}, Ref{MemRef}, Ref{MemRef}), $MemRef(A), $MemRef(B), $MemRef(C))


