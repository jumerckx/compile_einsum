using InteractiveUtils
macro code_ircode(ex0...)
    thecall = InteractiveUtils.gen_call_with_extracted_types_and_kwargs(@__MODULE__, :(Base.code_ircode), ex0)
    quote
        local results = $thecall
        length(results) == 1 ? results[1] : results
    end
end

using MLIR: IR, API

function registerAllDialects!()
    ctx = IR.context()
    registry = API.mlirDialectRegistryCreate()
    API.mlirRegisterAllDialects(registry)
    # handle = API.mlirGetDialectHandle__jlir__()
    # API.mlirDialectHandleInsertDialect(handle, registry)
    API.mlirContextAppendDialectRegistry(ctx, registry)
    API.mlirDialectRegistryDestroy(registry)

    API.mlirContextLoadAllAvailableDialects(ctx)
    return registry
end

function lowerModuleToLLVM(mod::IR.MModule)
    pm = IR.PassManager()

    IR.add_owned_pass!(pm, API.mlirCreateConversionConvertAffineForToGPU())
    # IR.add_owned_pass!(pm, API.mlirCreateConversionConvertAffineToStandard())
    # IR.add_owned_pass!(pm, API.mlirCreateConversionSCFToControlFlow())
    # IR.add_owned_pass!(pm, API.mlirCreateConversionFinalizeMemRefToLLVMConversionPass())
    # IR.add_owned_pass!(pm, API.mlirCreateConversionConvertFuncToLLVMPass())
    # IR.add_owned_pass!(pm, API.mlirCreateConversionArithToLLVMConversionPass())
    # IR.add_owned_pass!(pm, API.mlirCreateConversionConvertIndexToLLVMPass())
    # IR.add_owned_pass!(pm, API.mlirCreateConversionReconcileUnrealizedCasts())
    status = API.mlirPassManagerRunOnOp(pm, IR.get_operation(mod).operation)

    if status.value == 0
        error("Unexpected failure running pass failure")
    end
    return mod
end

function lowerModuleToNVVM(mod::IR.MModule)
    pm = IR.PassManager()

    pm_func = API.mlirPassManagerGetNestedUnder(pm, "func.func")

    API.mlirOpPassManagerAddOwnedPass(pm_func, API.mlirCreateConversionConvertNVGPUToNVVMPass())

    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateGPUGpuKernelOutlining())
    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateConversionFinalizeMemRefToLLVMConversionPass())
    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateConversionConvertFuncToLLVMPass())

    API.mlirOpPassManagerAddOwnedPass(pm_func, API.mlirCreateConversionConvertIndexToLLVMPass())
    API.mlirOpPassManagerAddOwnedPass(pm_func, API.mlirCreateConversionArithToLLVMConversionPass())

    pm_gpu = API.mlirPassManagerGetNestedUnder(pm, "gpu.module")

    API.mlirOpPassManagerAddOwnedPass(pm_gpu, API.mlirCreateTransformsStripDebugInfo())
    API.mlirOpPassManagerAddOwnedPass(pm_gpu, API.mlirCreateConversionConvertIndexToLLVMPass())
    API.mlirOpPassManagerAddOwnedPass(pm_gpu, API.mlirCreateConversionArithToLLVMConversionPass())
    API.mlirOpPassManagerAddOwnedPass(pm_gpu, API.mlirCreateConversionConvertGpuOpsToNVVMOps())
    API.mlirOpPassManagerAddOwnedPass(pm_gpu, API.mlirCreateConversionConvertNVVMToLLVMPass())
    API.mlirOpPassManagerAddOwnedPass(pm_gpu, API.mlirCreateConversionReconcileUnrealizedCasts())

    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateGPUGpuNVVMAttachTarget())
    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateConversionGpuToLLVMConversionPass())
    
    
    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateGPUGpuModuleToBinaryPass())

    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateConversionConvertIndexToLLVMPass())
    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateConversionConvertFuncToLLVMPass())

    API.mlirPassManagerAddOwnedPass(pm, API.mlirCreateConversionReconcileUnrealizedCasts())

    status = API.mlirPassManagerRunOnOp(pm, IR.get_operation(mod))

    if status.value == 0
        error("Unexpected failure running pass failure")
    end
    return mod
end
  

function jit(mod::IR.MModule; opt=0)
    paths = Base.unsafe_convert.(Ref(API.MlirStringRef), ["/storage/jumerckx/llvm_install_debug/lib/libmlir_cuda_runtime.so", "/storage/jumerckx/llvm_install_debug/lib/libmlir_runner_utils.so"])
    jit = API.mlirExecutionEngineCreate(
        mod,
        opt,
        length(paths), # numPaths
        paths, # libPaths
        true # enableObjectDump
    )
    function lookup(name)
        addr = API.mlirExecutionEngineLookup(jit, name)
        (addr == C_NULL) && error("Lookup failed.")
        return addr
    end
    return lookup
end

using StaticArrays

struct MemRef{T,N}
    allocated_pointer::Ptr{T}
    aligned_pointer::Ptr{T}
    offset::Int
    sizes::SVector{N, Int}
    strides::SVector{N, Int}
    data::Array{T, N}
end

function MemRef(a::Array{T,N}) where {T,N}
    @assert isbitstype(T) "non-isbitstype might not work"
    allocated_pointer = aligned_pointer = Base.unsafe_convert(Ptr{T}, a)
    offset = 0
    sizes = SVector{N}(collect(size(a)))
    strides = SVector{N}([1, cumprod(sizes)[1:end-1]...])

    return MemRef{T,N}(
        allocated_pointer,
        aligned_pointer,
        offset,
        sizes,
        strides,
        a,
    )
end