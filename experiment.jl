using MLIR, MLIR_jll
includet("utils.jl")
includet("Brutus.jl")

using Einsum, BenchmarkTools, MLIR

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

new_intrinsic = ()->Base.compilerbarrier(:const, error("Intrinsics should be compiled to MLIR!"))
@noinline for_(start, stop) where {I <: Integer} =  new_intrinsic()::Int

# a[i] = b[i, j] * c[j]
function example1(a::A, b::A, c::A) where {T, N, A<:AbstractArray{T, N}}
    i = for_(1, min(size(a, 1), size(b, 1)))
    temp = 0
    j = for_(1, size(b, 2))
    temp += max(T(0), b[i, j] * c[j])
end
@code_ircode example1(a, b, c)

# a[i, j] = b[i, k] * c[k, j]
function example2(a::A, b::A, c::A) where {T, N, A<:AbstractArray{T, N}}
    i = for_(1, min(size(a, 1), size(b, 1)))
    j = for_()
end



