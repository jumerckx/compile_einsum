using MLIR, MLIR_jll
includet("utils.jl")
includet("Brutus.jl")

using Einsum, MLIR, MacroTools

using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

Base.size(A::Brutus.MemRef) = Tuple(A.sizes)


Base.getindex(A::Brutus.MemRef{T}, I::Union{Integer, CartesianIndex}...) where {T} = Brutus.mlir_load(A, (I .- 1)...)
Base.getindex(A::Brutus.MemRef{T}, i1::Integer) where {T} = Brutus.mlir_load(A, Brutus.delinearize_index(i1 - 1, A)...)

Base.setindex!(A::Brutus.MemRef{T}, v, I::Union{Integer, CartesianIndex}...) where {T} = Brutus.mlir_store!(A, v, (I .- 1)...)
Base.setindex!(A::Brutus.MemRef{T}, v, i1::Integer) where {T} = Brutus.mlir_store!(A, v, Brutus.delinearize_index(i1 - 1, A)...)

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

ir = Base.code_ircode(example2, Tuple{Brutus.MemRef{Float64, 1}, Brutus.MemRef{Float64, 1}, Brutus.MemRef{Float64, 1}})

op = Brutus.code_mlir(example2, Tuple{Brutus.MemRef{Float64, 1}, Brutus.MemRef{Float64, 1}, Brutus.MemRef{Float64, 1}})

IR.verify(op)

mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)
pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)

addr = jit(mod; opt=3)("_mlir_ciface_example2")

a = rand(10)
b = rand(10)
c = similar(a)
@ccall $addr(Brutus.MemRef(c)::Ref{Brutus.MemRef}, Brutus.MemRef(a)::Ref{Brutus.MemRef}, Brutus.MemRef(b)::Ref{Brutus.MemRef})::Int

a - b

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

ir = Base.code_ircode(matmul, Tuple{Brutus.MemRef{Float64, 2}, Brutus.MemRef{Float64, 2}, Brutus.MemRef{Float64, 2}}) |> only
op = Brutus.code_mlir(matmul, Tuple{Brutus.MemRef{Float64, 2}, Brutus.MemRef{Float64, 2}, Brutus.MemRef{Float64, 2}})

mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)
pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)

addr = jit(mod; opt=3)("_mlir_ciface_matmul")

a = rand(10, 10)
b = rand(10, 10)
c = similar(a, (size(a, 1), size(b, 2)))

@ccall $addr(Brutus.MemRef(a)::Ref{Brutus.MemRef}, Brutus.MemRef(b)::Ref{Brutus.MemRef}, Brutus.MemRef(c)::Ref{Brutus.MemRef})::Int

function vadd(a, b, c)
    for i in eachindex(c)
        c[i] += b[i] + a[i]
    end
    return 1
end

ir = Base.code_ircode(vadd, Tuple{Brutus.MemRef{Float64, 1}, Brutus.MemRef{Float64, 1}, Brutus.MemRef{Float64, 1}}) |> only
op = Brutus.code_mlir(vadd, Tuple{Brutus.MemRef{Float64, 1}, Brutus.MemRef{Float64, 1}, Brutus.MemRef{Float64, 1}})
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

@ccall $addr(Brutus.MemRef(a)::Ref{Brutus.MemRef}, Brutus.MemRef(b)::Ref{Brutus.MemRef}, Brutus.MemRef(c)::Ref{Brutus.MemRef})::Int


# ir = Base.code_ircode(matmul, Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}) |> only
# op = Brutus.code_mlir(matmul, Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}})

# b = IR.Block()
# v = IR.push_argument!(b, IR.MLIRType(typeof(rand(20, 4))), IR.Location())

# MLIR.Dialects.Brutus.MemRef.ExtractStridedMetadata(; location=IR.Location(),
#        base_buffer_=IR.MLIRType(typeof(rand(20, 4))),
#        offset_=IR.IndexType(),
#        sizes_=IR.MLIRType[IR.IndexType(), IR.IndexType()],
#        strides_=IR.MLIRType[IR.IndexType(), IR.IndexType()],
#        source_=v) |> IR.get_results
