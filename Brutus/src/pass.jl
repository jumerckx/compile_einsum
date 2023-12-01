function lowerModuleToLLVM(mod::IR.MModule)
    pm = IR.PassManager()
    IR.add_pipeline!(MLIR.IR.OpPassManager(pm), "normalize-memrefs")
    IR.add_pipeline!(MLIR.IR.OpPassManager(pm), "affine-expand-index-ops")
    IR.add_owned_pass!(pm, API.mlirCreateConversionConvertAffineToStandard())
    IR.add_owned_pass!(pm, API.mlirCreateConversionSCFToControlFlow())
    IR.add_owned_pass!(pm, API.mlirCreateConversionFinalizeMemRefToLLVMConversionPass())
    IR.add_owned_pass!(pm, API.mlirCreateConversionConvertFuncToLLVMPass())
    IR.add_owned_pass!(pm, API.mlirCreateConversionArithToLLVMConversionPass())
    IR.add_owned_pass!(pm, API.mlirCreateConversionConvertIndexToLLVMPass())
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

function simplify(op::IR.Operation)
    pm = IR.PassManager()
     
    IR.add_pipeline!(IR.OpPassManager(pm), "canonicalize")
    IR.add_pipeline!(IR.OpPassManager(pm), "cse")
    IR.add_pipeline!(IR.OpPassManager(pm), "canonicalize")
    status = API.mlirPassManagerRunOnOp(pm, op)

    if status.value == 0
        error("Unexpected failure running pass failure")
    end
    return op
end

function simplify(mod::IR.MModule)
    simplify(IR.get_operation(mod))
    return mod
end