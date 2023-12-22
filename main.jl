using MLIR, MLIR_jll
includet("utils.jl")
using Brutus
import Brutus: MemRef
using Meinsum, BenchmarkTools, MLIR, MacroTools

using MLIR.Dialects.Affine
using MLIR.IR


using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

prettify(Meinsum._meinsum(:(A[i] = B[i, j] * C[j])))

prettify(Meinsum._meinsum(:(A[i] = B[i] * C[i])))

prettify(Meinsum._meinsum(:(A[i, j] = B[i, k] * C[k, j])))

f(a, b, c) = @meinsum a[i] = b[i, j] * c[j];

a = zeros(10)
b = rand(10, 2)
c = rand(2)

@code_ircode f(Brutus.MemRef(a), Brutus.MemRef(b), Brutus.MemRef(c))

op = Brutus.@code_mlir f(Brutus.MemRef(a), Brutus.MemRef(b), Brutus.MemRef(c))

f(y, x) = @meinsum y[i] = x[i]
@code_ircode f(Brutus.MemRef(a), Brutus.MemRef(a))
Base.code_ircode(f, Tuple{Brutus.MemRef{Float64, 1}, Brutus.MemRef{Float64, 1}})
Brutus.@code_mlir f(Brutus.MemRef(a), Brutus.MemRef(a))
Brutus.code_mlir(f, Tuple{Brutus.MemRef{Float64, 1}, Brutus.MemRef{Float64, 1}}, do_simplify=false)


mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)
pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)
addr = jit(mod; opt=3)("_mlir_ciface_f")

@ccall $addr(MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef}, MemRef(c)::Ref{MemRef})::Int

@assert a ≈ sum(b .* c', dims=2)

##############################################################

f(a, b, c) = @meinsum a[i, j] = b[i, k] * c[k, j];
a = zeros(Float16, 3, 4)
b = rand(Float16, 3, 2)
c = rand(Float16, 2, 4)

prettify(@macroexpand @meinsum a[i, j] = b[i, k] * c[k, j])

@code_ircode f(Brutus.MemRef(a), Brutus.MemRef(b), Brutus.MemRef(c))

op = Brutus.@code_mlir f(Brutus.MemRef(a), Brutus.MemRef(b), Brutus.MemRef(c))

IR.verify(op)

mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)
pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)
addr = jit(mod; opt=3)("_mlir_ciface_f")

@ccall $addr(MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef}, MemRef(c)::Ref{MemRef})::Int

@assert a ≈ b * c