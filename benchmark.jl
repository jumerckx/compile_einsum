using MLIR, MLIR_jll
includet("utils.jl")
using Brutus
import Brutus: MemRef
using Einsum, Meinsum, BenchmarkTools, MLIR, MacroTools, LinearAlgebra, Tullio, LoopVectorization
using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

# a = rand(100, 100)
# b = rand(100, 70)
# c = rand(70, 100)

a = rand(1024, 1024)
b = rand(1024, 1024)
c = rand(1024, 1024)

function get_functions(expr, names)
    quote
        f_einsum($(names...),) = @einsum $expr
        f_mlir($(names...),) = @meinsum $expr

        (f_einsum, f_mlir)
    end
end
macro get_functions(expr, names)
    get_functions(expr, names.args)
end

first(@get_functions(y[i, j] = x[i, k] * z[k, j], (y, x, z)))(a, b, c)

function benchmark((f_einsum, f_mlir), sizes)
    args = []
    for s in sizes
        push!(args, rand(s...))
    end
    args_mlir = MemRef.(args)

    f_einsum(args...)
    @time f_einsum(args...)

    op = Brutus.@code_mlir f_mlir(args_mlir...)
    mod = IR.MModule(IR.Location())
    push!(IR.get_body(mod), op)
    pm = lowerModuleToLLVM(mod)
    op = IR.get_operation(mod)
    addr = jit(mod; opt=3)("_mlir_ciface_$(String(Symbol(f)))")
    f_mlir(args_mlir) = ccall(addr, Int, (Ref{MemRef}, Ref{MemRef}, Ref{MemRef}), args_mlir...)
    f_mlir(args_mlir)
    @time f_mlir(args_mlir)
    # @btime $f($x, $y, $z)
end

benchmark(@get_functions(y[i, j] = x[i, k] * z[k, j], (y, x, z)), ((100, 100), (100, 100), (100, 100)))

f_einsum(a, b, c) = @einsum a[i, j] += b[i, k] * c[k, j]

f_tullio(a, b, c) = @tullio a[i, j] = b[i, k] * c[k, j]

@benchmark f_einsum(a, b, c)

@code_llvm f_tullio(a, b, c)


function g(a, b, c)
    @meinsum a[i, j] = b[i, k] * c[k, j]
    @meinsum a[i, j] += b[i, j]
    return 1
end

@code_ircode g(MemRef(a), MemRef(b), MemRef(c))

Brutus.@code_mlir g(MemRef(a), MemRef(b), MemRef(c))

prettify(@macroexpand @meinsum a[i, j] = c[i, j] + b[i, k])

prettify(@macroexpand @meinsum a[i, j] = b[i, k] * c[k, j])

a = rand(3, 3)
b = rand(3, 10000)
c = rand(10000, 3)

@tullio a[i, j] = b[i, k] * c[k, j]