struct MemRef{T,N} <: DenseArray{T, N}
    allocated_pointer::Ptr{T}
    aligned_pointer::Ptr{T}
    offset::Int
    sizes::NTuple{N, Int}
    strides::NTuple{N, Int}
    data::Array{T, N}
end
import Base.show
Base.show(io::IO, A::Brutus.MemRef{T, N}) where {T, N} = print(io, "Brutus.MemRef{$T,$N} (size $(join(A.sizes, "×")))")
Base.show(io::IO, ::MIME{Symbol("text/plain")}, X::Brutus.MemRef) = show(io, X)

function MemRef(a::Array{T,N}) where {T,N}
    @assert isbitstype(T) "Array element type should be isbits, got $T."
    allocated_pointer = a.ref.mem.ptr
    aligned_pointer = a.ref.ptr_or_offset
    offset = Int((aligned_pointer - allocated_pointer)//sizeof(T))
    @assert offset == 0 "Arrays with Memoryref offset are, as of yet, unsupported."
    sizes = size(a)
    strides = Tuple([1, cumprod(sizes)[1:end-1]...])

    return MemRef{T,N}(
        allocated_pointer,
        aligned_pointer,
        offset,
        sizes,
        strides,
        a,
    )
end

Base.size(A::MemRef) = Tuple(A.sizes)

Base.getindex(A::MemRef{T}, I::Union{Integer, CartesianIndex}...) where {T} = mlir_load(A, (I .- 1)...)
Base.getindex(A::MemRef{T}, i1::Integer) where {T} = mlir_load(A, delinearize_index(i1 - 1, A)...)

Base.setindex!(A::MemRef{T}, v, I::Union{Integer, CartesianIndex}...) where {T} = mlir_store!(A, v, (I .- 1)...)
Base.setindex!(A::MemRef{T}, v, i1::Integer) where {T} = mlir_store!(A, v, delinearize_index(i1 - 1, A)...)
