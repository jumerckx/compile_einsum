using MLIR
includet("utils.jl")
using Brutus
import Brutus: MemRef
using Meinsum, BenchmarkTools, MLIR, MacroTools

using MLIR.Dialects.Affine
using MLIR.IR

using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses() # to be fixed: this should only be run once each session, otherwise it'll segfault!
API.mlirRegisterAllLLVMTranslations(ctx.context)

### Simple vector copy example ###

# The Meinsum package is a copy of Einsum.jl where the loops that are generated are replaced by `begin_for` and `yield_for`.
Meinsum._meinsum(:(y[i] = x[i]))

# We can put this in a function:
f(x, y) = @meinsum y[i] = x[i];

x = rand(10);
y = similar(x);

# Have a look at the Julia IR:
@code_ircode f(Brutus.MemRef(x), Brutus.MemRef(y))

# And now, generate MLIR:
op = Brutus.@code_mlir f(Brutus.MemRef(x), Brutus.MemRef(y))

@assert IR.verify(op)

# To run the code, we need to first:
#   put the operation in a MLIR module:
mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)

#   lower all MLIR dialects to the llvm dialect:
pm = lowerModuleToLLVM(mod)

#   retrieve the operation from the module:
op = IR.get_operation(mod)

#   jit the function using an execution engine:
addr = jit(mod; opt=3)("_mlir_ciface_f")

# Finally, we call the function like a regular C-function:
@ccall $addr(MemRef(x)::Ref{MemRef}, MemRef(y)::Ref{MemRef})::Int

@assert y == x

### Matmul example ###

# This generates too few blocks (two yields in after each other) and would
# require a more robust way of representing loops in IR:
# f(A, B, C; α=eltype(A)(1)) = @meinsum C[i, j] = α * (A[i, k] * B[k, j])

# but this works:
f(A, B, C) = @meinsum C[i, j] =A[i, k] * B[k, j]

A, B = rand(3, 4), rand(4, 3)
C = similar(A, 3, 3)

@code_ircode f(Brutus.MemRef(A), Brutus.MemRef(B), Brutus.MemRef(C))
op = Brutus.@code_mlir f(Brutus.MemRef(A), Brutus.MemRef(B), Brutus.MemRef(C))

@assert IR.verify(op)

mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)
pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)
addr = jit(mod; opt=3)("_mlir_ciface_f")
@ccall $addr(MemRef(A)::Ref{MemRef}, MemRef(B)::Ref{MemRef}, MemRef(C)::Ref{MemRef})::Int

@assert C ≈ A*B
