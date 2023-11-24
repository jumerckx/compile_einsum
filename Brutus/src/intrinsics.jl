new_intrinsic = ()->Base.compilerbarrier(:const, error("Intrinsics should be compiled to MLIR!"))

@noinline begin_for(start::I, stop::I) where {I <: Integer} =  new_intrinsic()::Int
@noinline begin_for(result::T, start::I, stop::I) where {I, T} = new_intrinsic()::Tuple{Int, T}

@noinline yield_for(val::T=nothing) where T = new_intrinsic()::T

@noinline mlir_load(A::Brutus.MemRef{T}, I::Union{Integer, CartesianIndex}...) where T = Brutus.new_intrinsic()::T
@noinline mlir_store!(A::Brutus.MemRef{T}, v, I::Union{Integer, CartesianIndex}...) where {T} = new_intrinsic()::T

@noinline delinearize_index(i1::Integer, basis::NTuple{N, Int}) where {N} = Brutus.new_intrinsic()::NTuple{N, Int}
delinearize_index(i1::Integer, A::Brutus.MemRef) = delinearize_index(i1, size(A))
