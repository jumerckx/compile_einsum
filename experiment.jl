using MLIR, MLIR_jll
includet("utils.jl")
includet("Brutus.jl")

using Einsum, MLIR, MacroTools

using MLIR.Dialects.Affine
using MLIR.IR


using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

b = rand(1000, 2000);
c = rand(2000);
a = similar(b, (1000, ));


# a[i] = b[i, j] * c[j]
@inbounds function example1(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}) where {T}
    i = Brutus.begin_for(1, size(a, 1))
    @goto body1_begin
    @label body1_begin
        j, accum = Brutus.begin_for(T(0), 1, size(b, 2))
        @goto body2_begin
            @label body2_begin
                accum += b[i, j] * c[j]
            accum = Brutus.yield_for(accum)
            @goto body2_end
        @label body2_end
        a[i] = accum
        Brutus.yield_for()
        @goto body1_end
    @label body1_end
    return 1
end
@show Base.code_ircode(example1, Tuple{Vector{Float64}, Matrix{Float64}, Vector{Float64}})

# a[i] = a[i] + b[i]
Base.@propagate_inbounds function example2(a::AbstractVector{T}, b::AbstractVector{T}) where {T}
    i = Brutus.begin_for(1, size(a, 1))
    @goto body1_begin
        @label body1_begin
        Base.@inbounds a[i] += b[i]
        Brutus.yield_for()
        @goto body1_end
    @label body1_end
    return 1
end
ir = Base.code_ircode(example2, Tuple{Vector{Float64}, Vector{Float64}})

op = Brutus.@code_mlir example2(Brutus.MemRef(c), Brutus.MemRef(c))

mod = parse(IR.MModule, """module {
  func.func @S1(%arg0: memref<?xf64, strided<[1], offset: ?>>, %arg1: memref<?x?xf64, strided<[1, ?], offset: ?>>, %arg2: memref<?xf64, strided<[1], offset: ?>>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) {
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f64
    %0 = index.add %arg5, %c1
    %1 = affine.for %arg7 = 1 to %0 iter_args(%arg8 = %cst) -> (f64) {
      %6 = index.casts %arg3 : index to i64
      %7 = arith.subi %6, %c1_i64 : i64
      %8 = index.casts %arg7 : index to i64
      %9 = arith.subi %8, %c1_i64 : i64
      %10 = index.casts %7 : i64 to index
      %11 = index.casts %9 : i64 to index
      %12 = memref.load %arg1[%10, %11] : memref<?x?xf64, strided<[1, ?], offset: ?>>
      %13 = affine.delinearize_index %11 into (%arg6) : index
      %14 = memref.load %arg2[%13] : memref<?xf64, strided<[1], offset: ?>>
      %15 = arith.mulf %12, %14 : f64
      %16 = arith.addf %arg8, %15 : f64
      affine.yield %16 : f64
    }
    %2 = index.casts %arg3 : index to i64
    %3 = arith.subi %2, %c1_i64 : i64
    %4 = index.casts %3 : i64 to index
    %5 = affine.delinearize_index %4 into (%arg4) : index
    memref.store %1, %arg0[%5] : memref<?xf64, strided<[1], offset: ?>>
    return
  }
  func.func @example1(%arg0: memref<?xf64, strided<[1], offset: ?>>, %arg1: memref<?x?xf64, strided<[1, ?], offset: ?>>, %arg2: memref<?xf64, strided<[1], offset: ?>>) -> i64 attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64, strided<[1], offset: ?>>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf64, strided<[1, ?], offset: ?>>
    %dim_1 = memref.dim %arg2, %c0 : memref<?xf64, strided<[1], offset: ?>>
    %0 = index.add %dim, %c1
    affine.for %arg3 = 1 to %0 {
      func.call @S1(%arg0, %arg1, %arg2, %arg3, %dim, %dim_0, %dim_1) : (memref<?xf64, strided<[1], offset: ?>>, memref<?x?xf64, strided<[1, ?], offset: ?>>, memref<?xf64, strided<[1], offset: ?>>, index, index, index, index) -> ()
    }
    return %c1_i64 : i64
  }
}
""")

op = Brutus.@code_mlir example1(Brutus.MemRef(a), Brutus.MemRef(b), Brutus.MemRef(c))

# IR.verify(op) # currently fails because loop body has more than one block.

pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)

addr = jit(mod; opt=3)("_mlir_ciface_example1")

@ccall $addr(MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef}, MemRef(c)::Ref{MemRef})::Int


a ≈ sum(b .* c', dims=2)