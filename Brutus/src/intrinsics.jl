new_intrinsic = ()->Base.compilerbarrier(:const, error("Intrinsics should be compiled to MLIR!"))

struct MLIRIndex <: Integer
    i::Int
end
IR.MLIRType(::Type{MLIRIndex}) = IR.IndexType()

@noinline mlir_indexadd(a::Integer, b::Integer) = new_intrinsic()::MLIRIndex
@noinline mlir_indexsub(a::Integer, b::Integer) = new_intrinsic()::MLIRIndex
@noinline mlir_indexmul(a::Integer, b::Integer) = new_intrinsic()::MLIRIndex
@noinline mlir_indexdiv(a::Integer, b::Integer) = new_intrinsic()::MLIRIndex

import Base: +, -, *, //
for (f, intrinsic) in [
        (:+, Brutus.mlir_indexadd),
        (:-, Brutus.mlir_indexsub),
        (:*, Brutus.mlir_indexmul),
        (://, Brutus.mlir_indexdiv) # TODO: not sure about this one
    ]
    @eval begin
        $f(a::MLIRIndex, b::Integer) = $intrinsic(a, b)
        $f(a::Integer, b::MLIRIndex) = $intrinsic(a, b)
    end
end

@noinline begin_for(start::Integer, stop::Integer) =  new_intrinsic()::MLIRIndex
@noinline begin_for(result::T, start::Integer, stop::Integer) where T = new_intrinsic()::Tuple{MLIRIndex, T}

@noinline yield_for(val::T=nothing) where T = new_intrinsic()::T

struct MemRef{T,N} <: DenseArray{T, N}
    allocated_pointer::Ptr{T}
    aligned_pointer::Ptr{T}
    offset::Int
    sizes::NTuple{N, MLIRIndex}
    strides::NTuple{N, Int}
    data::Array{T, N}
end
function MemRef(a::Array{T,N}) where {T,N}
    @assert isbitstype(T) "Array element type should be isbits, got $T."
    allocated_pointer = a.ref.mem.ptr
    aligned_pointer = a.ref.ptr_or_offset
    offset = Int((aligned_pointer - allocated_pointer)//sizeof(T))
    @assert offset == 0 "Arrays with Memoryref offset are, as of yet, unsupported."
    sizes = size(a)
    strides = Tuple([1, cumprod(size(a))[1:end-1]...])
    
    return MemRef{T,N}(
        allocated_pointer,
        aligned_pointer,
        offset,
        sizes,
        strides,
        a,
        )
end

Base.show(io::IO, A::Brutus.MemRef{T, N}) where {T, N} = print(io, "Brutus.MemRef{$T,$N} (size $(join(A.sizes, "×")))")
Base.show(io::IO, ::MIME{Symbol("text/plain")}, X::Brutus.MemRef) = show(io, X)
Base.size(A::MemRef) = A.sizes

@noinline mlir_load(A::Brutus.MemRef{T}, I::Union{Integer, CartesianIndex}...) where T = Brutus.new_intrinsic()::T
@noinline mlir_store!(A::Brutus.MemRef{T}, v::T, I::Union{Integer, CartesianIndex}...) where {T} = new_intrinsic()::T

Base.getindex(A::MemRef{T}, I::Union{Integer, CartesianIndex}...) where {T} = mlir_load(A, (I .- 1)...)
Base.getindex(A::MemRef{T}, i1::Integer) where {T} = mlir_load(A, delinearize_index(i1 - 1, A)...)

Base.setindex!(A::MemRef{T}, v, I::Union{Integer, CartesianIndex}...) where {T} = mlir_store!(A, T(v), (I .- 1)...)
Base.setindex!(A::MemRef{T}, v, i1::Integer) where {T} = mlir_store!(A, T(v), delinearize_index(i1 - 1, A)...)

"""
This intrinsic will be mapped to the MLIR `affine.delinearize_index` operation.

Note that this operation assumes zero-based indexing and row-major layout of the basis. 
By providing a `MemRef` instead, the conversion between column-major and row-major layout
is handled automatically, the index should still be zero-based.
"""
@noinline delinearize_index(i1::Integer, basis::NTuple{N, MLIRIndex}) where {N} = Brutus.new_intrinsic()::NTuple{N, MLIRIndex}
delinearize_index(i1::Integer, A::Brutus.MemRef) = reverse(delinearize_index(i1, reverse(size(A))))
