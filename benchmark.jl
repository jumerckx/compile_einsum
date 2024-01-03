using MLIR
includet("utils.jl")
using Brutus
import Brutus: MemRef
using Einsum, Meinsum, BenchmarkTools, MLIR, MacroTools, LinearAlgebra, Tullio, LoopVectorization
using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

a = rand(1024, 1024)
b = rand(1024, 1024)
c = rand(1024, 1024)

f_einsum(a, b, c) = @einsum a[i, j] = b[i, k] * c[k, j]

f_tullio(a, b, c) = @tullio a[i, j] = b[i, k] * c[k, j]

f_tullio_no_avx(a, b, c) = @tullio avx=false a[i, j] = b[i, k] * c[k, j]

f_tullio_no_threads_fastmath(a, b, c) = @tullio threads=false fastmath=false a[i, j] = b[i, k] * c[k, j]

f_tullio_no_threads_avx_fastmath(a, b, c) = @tullio threads=false avx=false fastmath=false a[i, j] = b[i, k] * c[k, j]

f_tullio_no_avx_fastmath(a, b, c) = @tullio avx=false fastmath=false a[i, j] = b[i, k] * c[k, j]

_f_mlir(a, b, c) = @meinsum a[i, j] = b[i, k] * c[k, j]
op = Brutus.@code_mlir _f_mlir(MemRef(a), MemRef(b), MemRef(c))
mod = IR.MModule(IR.Location())
push!(IR.get_body(mod), op)
pm = lowerModuleToLLVM(mod)
op = IR.get_operation(mod)
addr = jit(mod; opt=3)("_mlir_ciface__f_mlir")
@ccall $addr(MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef}, MemRef(c)::Ref{MemRef})::Int
f_mlir(a, b, c) = @ccall $addr(MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef}, MemRef(c)::Ref{MemRef})::Int

@benchmark f_einsum(a, b, c)
# BenchmarkTools.Trial: 2 samples with 1 evaluation.
#  Range (min … max):  3.417 s …   3.443 s  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     3.430 s              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   3.430 s ± 18.389 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%
#  Memory estimate: 0 bytes, allocs estimate: 0.

@benchmark f_tullio_no_avx(a, b, c)
# BenchmarkTools.Trial: 5 samples with 1 evaluation.
#  Range (min … max):  1.032 s …  1.037 s  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     1.037 s             ┊ GC (median):    0.00%
#  Time  (mean ± σ):   1.035 s ± 2.910 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%
#  Memory estimate: 0 bytes, allocs estimate: 0.

@benchmark f_tullio_no_threads_fastmath(a, b, c)
# BenchmarkTools.Trial: 43 samples with 1 evaluation.
#  Range (min … max):  110.287 ms … 137.943 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     117.374 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   117.971 ms ±   5.339 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%
#  Memory estimate: 0 bytes, allocs estimate: 0.

@benchmark f_tullio_no_threads_avx_fastmath(a, b, c)
# Range (min … max):  3.501 s …  3.513 s  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     3.507 s             ┊ GC (median):    0.00%
#  Time  (mean ± σ):   3.507 s ± 8.616 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%
#  Memory estimate: 0 bytes, allocs estimate: 0.

@benchmark f_tullio_no_avx_fastmath(a, b, c)
# BenchmarkTools.Trial: 5 samples with 1 evaluation.
#  Range (min … max):  1.119 s …    1.664 s  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     1.124 s               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   1.300 s ± 253.190 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%
#  Memory estimate: 0 bytes, allocs estimate: 0.

@benchmark f_tullio(a, b, c)
# BenchmarkTools.Trial: 75 samples with 1 evaluation.
#  Range (min … max):  63.138 ms … 86.292 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     66.770 ms              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   67.157 ms ±  2.840 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%
# Memory estimate: 0 bytes, allocs estimate: 0.

@benchmark f_mlir(a, b, c)
# BenchmarkTools.Trial: 2 samples with 1 evaluation.
#  Range (min … max):  3.410 s …   3.446 s  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     3.428 s              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   3.428 s ± 25.443 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%
#  Memory estimate: 1.08 KiB, allocs estimate: 30.