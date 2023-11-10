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

b = rand(1000, 2000);
c = rand(2000);
a = similar(b, (1000, ));

new_intrinsic = ()->Base.compilerbarrier(:const, error("Intrinsics should be compiled to MLIR!"))
@noinline begin_for(start::I, stop::I) where {I <: Integer} =  new_intrinsic()::Int
@noinline begin_for(result::T, start::I, stop::I) where {I, T} = new_intrinsic()::Tuple{T, Int}

@noinline yield_for(val::T=nothing) where T = new_intrinsic()::T

# a[i] = b[i, j] * c[j]
function example1(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}) where {T}
    i = begin_for(1, size(a, 1))
    @goto body1
    @label body1
        accum, j = begin_for(T(0), 1, size(b, 2))
        @goto body2
            @label body2
                accum += b[i, j] * c[j]
            accum = yield_for(accum)
        a[i] = accum
        yield_for()
    return nothing
end
Base.code_ircode(example1, Tuple{Vector{Float64}, Matrix{Float64}, Vector{Float64}})

function example2(a::AbstractVector{T}, b::AbstractVector{T}) where {T}
    i = begin_for(1, size(a, 1))
    @goto body1
    @label body1
        a[i] += b[i]
        yield_for()
end
Base.code_ircode(example2, Tuple{Vector{Float64}, Vector{Float64}})

# Brutus.@code_mlir example1(a, b, c,)
