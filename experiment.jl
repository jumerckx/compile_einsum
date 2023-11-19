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

function example2(a::AbstractVector{T}, b::AbstractVector{T}) where {T}
    i = Brutus.begin_for(1, size(a, 1))
    @goto body1_begin
        @label body1_begin
        a[i] += b[i]
        Brutus.yield_for()
        @goto body1_end
    @label body1_end
end
ir = Base.code_ircode(example2, Tuple{Vector{Float64}, Vector{Float64}})

# @show Brutus.@code_mlir example2(c, c)

op = Brutus.@code_mlir example1(a, b, c)

IR.verify(op) # currently fails because loop body has more than one block.

