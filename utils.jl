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

function mlir_opt(mod::IR.MModule, pipeline::String)
    pm = IR.PassManager()
    IR.add_pipeline!(MLIR.IR.OpPassManager(pm), pipeline)
    status = API.mlirPassManagerRunOnOp(pm, IR.get_operation(mod).operation)
    if status.value == 0
        error("Unexpected failure running pass failure")
    end
    return mod
end

function lowerModuleToLLVM(mod::IR.MModule)
    pm = IR.PassManager()

    IR.add_pipeline!(
        MLIR.IR.OpPassManager(pm), 
        "func.func(convert-vector-to-scf{full-unroll=false lower-tensors=false target-rank=1}),func.func(convert-linalg-to-loops),lower-affine,convert-scf-to-cf,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},cse,convert-vector-to-llvm{enable-amx=false enable-arm-neon=false enable-arm-sme=false enable-arm-sve=false enable-x86vector=false force-32bit-vector-indices=true reassociate-fp-reductions=false use-opaque-pointers=true},func.func(convert-math-to-llvm{approximate-log1p=true}),expand-strided-metadata,lower-affine,finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false use-opaque-pointers=true},convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false use-opaque-pointers=true},convert-index-to-llvm{index-bitwidth=0},reconcile-unrealized-casts"
    )

    # IR.add_owned_pass!(pm, API.mlirCreateConversionConvertAffineToStandard())
    # IR.add_owned_pass!(pm, API.mlirCreateConversionConvertVectorToLLVMPass())
    # IR.add_pipeline!(MLIR.IR.OpPassManager(pm), "normalize-memrefs")
    # IR.add_pipeline!(MLIR.IR.OpPassManager(pm), "affine-expand-index-ops")
    # IR.add_owned_pass!(pm, API.mlirCreateConversionSCFToControlFlow())
    # IR.add_owned_pass!(pm, API.mlirCreateConversionFinalizeMemRefToLLVMConversionPass())
    # IR.add_owned_pass!(pm, API.mlirCreateConversionConvertFuncToLLVMPass())
    # IR.add_owned_pass!(pm, API.mlirCreateConversionArithToLLVMConversionPass())
    # IR.add_owned_pass!(pm, API.mlirCreateConversionConvertIndexToLLVMPass())
    IR.add_owned_pass!(pm, API.mlirCreateConversionReconcileUnrealizedCasts())
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
    paths = Base.unsafe_convert.(Ref(API.MlirStringRef), ["/storage/jumerckx/llvm_install_debug/lib/libmlir_cuda_runtime.so", "/storage/jumerckx/llvm_install_debug/lib/libmlir_runner_utils.so", "/storage/jumerckx/llvm_install_debug/lib/libmlir_c_runner_utils.so"])
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