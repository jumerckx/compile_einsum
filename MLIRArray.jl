using MLIR, MLIR_jll
includet("utils.jl")
# includet("Brutus.jl")
using Brutus

using Einsum, MLIR, MacroTools

import Brutus: MemRef

using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

@inbounds function example1(a::AbstractArray{T}, b::AbstractArray{T}, c::AbstractArray{T}) where {T}
    i = Brutus.begin_for(1, size(a, 1)+1)
    @goto body1_begin
    @label body1_begin
        j, accum = Brutus.begin_for(T(0), 1, size(b, 2))
        @goto body2_begin
            @label body2_begin
                accum += b[i, j] * c[j]
            @goto yield_block_1
            @label yield_block_1
            accum = Brutus.yield_for(accum)
            @goto body2_end
        @label body2_end
        a[i] = accum
        @goto yield_block_2
        @label yield_block_2
        Brutus.yield_for()
        @goto body1_end
    @label body1_end
    return 1
end
ir = Base.code_ircode(example1, Tuple{MemRef{Float64, 1}, MemRef{Float64, 2}, MemRef{Float64, 1}})
op = Brutus.code_mlir(example1, Tuple{MemRef{Float64, 1}, MemRef{Float64, 2}, MemRef{Float64, 1}})

IR.verify(op)

mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)

b = rand(1000, 2000);
c = rand(2000);
a = similar(b, (1000, ));

pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)

addr = jit(mod; opt=3)("_mlir_ciface_example1")

@ccall $addr(MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef}, MemRef(c)::Ref{MemRef})::Int

a ≈ sum(b .* c', dims=2)

# note:
a ≈ sum(b[:, begin:end-1] .* c[begin:end-1]', dims=2)

a
sum(b .* c', dims=2)

function example2(c::AbstractVector{T}, a::AbstractVector{T}, b::AbstractVector{T}) where {T}
    i = Brutus.begin_for(1, size(a, 1))
    @goto body1_begin
        @label body1_begin
        Base.@inbounds c[i] =  a[i] * b[i]
        Brutus.yield_for()
        @goto body1_end
    @label body1_end
    return 1
end

ir = Base.code_ircode(example2, Tuple{MemRef{Float64, 1}, MemRef{Float64, 1}, MemRef{Float64, 1}})

op = Brutus.code_mlir(example2, Tuple{MemRef{Float64, 1}, MemRef{Float64, 1}, MemRef{Float64, 1}})

op

IR.verify(op)

mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)
pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)

addr = jit(mod; opt=3)("_mlir_ciface_example2")

a = rand(10)
b = rand(10)
c = similar(a)
@ccall $addr(MemRef(c)::Ref{MemRef}, MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef})::Int

c ≈ a .* b

function matmul(a::A, b::A, c::A) where {A <: AbstractMatrix{T}} where T
    for I in eachindex(IndexCartesian(), c)
        i, j = Tuple(I)
        temp = eltype(c)(0)
        for k in axes(a, 2)
            temp += a[i, k] * b[k, j]
        end
        c[i, j] = temp
    end
    return 1
end

ir = Base.code_ircode(matmul, Tuple{MemRef{Float64, 2}, MemRef{Float64, 2}, MemRef{Float64, 2}}) |> only
op = Brutus.code_mlir(matmul, Tuple{MemRef{Float64, 2}, MemRef{Float64, 2}, MemRef{Float64, 2}})

mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)
pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)

addr = jit(mod; opt=3)("_mlir_ciface_matmul")

a = rand(10, 10)
b = rand(10, 10)
c = similar(a, (size(a, 1), size(b, 2)))

@ccall $addr(MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef}, MemRef(c)::Ref{MemRef})::Int

c ≈ a*b

function vadd(a, b, c)
    for i in eachindex(c)
        c[i] += b[i] + a[i]
    end
    return 1
end

ir = Base.code_ircode(vadd, Tuple{MemRef{Float64, 1}, MemRef{Float64, 1}, MemRef{Float64, 1}}) |> only
op = Brutus.code_mlir(vadd, Tuple{MemRef{Float64, 1}, MemRef{Float64, 1}, MemRef{Float64, 1}})
IR.verify(op)

op = Brutus.code_mlir(vadd, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}})
IR.verify(op)

mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)
pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)

addr = jit(mod; opt=3)("_mlir_ciface_vadd")

@show op

a = rand(10)
b = rand(10)
c = zeros(size(a))

@ccall $addr(MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef}, MemRef(c)::Ref{MemRef})::Int
