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

prettify(Einsum._einsum(:(A[i] = B[i] * C[i])))

prettify(Einsum._einsum(:(A[i, j] = B[i, k] * C[k, j])))

f(a, b, c) = @meinsum a[i] = b[i, j] * c[j];

a = zeros(10)
b = rand(10, 2)
c = rand(2)

@code_ircode f(Brutus.MemRef(a), Brutus.MemRef(b), Brutus.MemRef(c))

op = Brutus.@code_mlir f(Brutus.MemRef(a), Brutus.MemRef(b), Brutus.MemRef(c))

mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)
pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)
addr = jit(mod; opt=3)("_mlir_ciface_f")

@ccall $addr(MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef}, MemRef(c)::Ref{MemRef})::Int

@assert a ≈ sum(b .* c', dims=2)

##############################################################

f(a, b, c) = @meinsum a[i, j] = sin(b[i, k] - c[k, j]);
a = zeros(3, 4)
b = rand(3, 2)
c = rand(2, 4)

@code_ircode f(Brutus.MemRef(a), Brutus.MemRef(b), Brutus.MemRef(c))

op = Brutus.@code_mlir f(Brutus.MemRef(a), Brutus.MemRef(b), Brutus.MemRef(c))

mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)
pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)
addr = jit(mod; opt=3)("_mlir_ciface_f")

@ccall $addr(MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef}, MemRef(c)::Ref{MemRef})::Int

@assert a ≈ b * c