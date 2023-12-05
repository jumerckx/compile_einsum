using MLIR, MLIR_jll
includet("utils.jl")
using Brutus
import Brutus: MemRef
using Einsum, Meinsum, BenchmarkTools, MLIR, MacroTools, LinearAlgebra
using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

a = rand(100, 100)
b = rand(100, 70)
c = rand(70, 100)

function f(a, b, c)
    @einsum a[i, j] = b[i, k] * c[k, j];
    nothing
end
f(a, b, c)

function _einsum_function(expr, names)
    quote
        ($(names...),) -> @einsum $expr
    end
end
macro einsum_function(expr, names)
    @show names
    _einsum_function(expr, names.args)
end

_einsum_function(:(y[i, j] = x[i, k] * z[k, j]), (:y, :x, :z))

test = @einsum_function(y[i, j] = x[i, k] * z[k, j], (y, x, z))(a, b, c)

function _mlir_function(expr, names, sizes)
    :(Brutus.MemRef())
    quote
        _f = ($(names...)) -> @meinsum $expr
        Brutus.@code_mlir _f(Brutus.MemRef($sizes[1]), Brutus.MemRef($sizes[2]), Brutus.MemRef($sizes[3]))
        ($(names...),) -> @meinsum $expr
    end
    end
end

f_mlir(a, b, c) = @meinsum a[i, j] = b[i, k] * c[k, j]
function get_mlir_function(f, a, b, c)
    op = Brutus.@code_mlir f(Brutus.MemRef(a), Brutus.MemRef(b), Brutus.MemRef(c))

    mod = IR.MModule(IR.Location())
    push!(IR.get_body(mod), op)
    pm = lowerModuleToLLVM(mod)
    op = IR.get_operation(mod)
    addr = jit(mod; opt=3)("_mlir_ciface_$(Symbol(f))")

    return (a, b, c)->ccall(addr, Int, (Ref{MemRef}, Ref{MemRef}, Ref{MemRef}), MemRef(a), MemRef(b), MemRef(c))

end

@benchmark $(get_mlir_function(f_mlir, a, b, c))(a, b, c)

function _time_einsum(expr, sizes::Dict{Symbol, NTuple{N, Int}}) where N
    exprs = []
    for (varname, size) in sizes
        push!(exprs, :($varname = rand($size)))
    end
    @warn Expr(:block, exprs...)

    quote
        $(Expr(:block, exprs...))
        @time @einsum $expr
    end
end

macro time_einsum(ex)
     _time_einsum(ex.args...)
end

@time_einsum (:(A[i, j] = B[i, k] * C[k, j]), Dict([:A=>(100, 100), :B=>(100, 70), :C=>(70, 100)]))