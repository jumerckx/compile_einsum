module Brutus

__revise_mode__ = :eval

import LLVM
using MLIR.IR
using MLIR: API
using MLIR.Dialects: arith, func, cf, std, Arith, Memref, Index, Builtin, Affine
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode

IR.MLIRType(::Type{Nothing}) = IR.MLIRType(API.mlirLLVMVoidTypeGet(IR.context()))

new_intrinsic = ()->Base.compilerbarrier(:const, error("Intrinsics should be compiled to MLIR!"))
@noinline begin_for(start::I, stop::I) where {I <: Integer} =  new_intrinsic()::Int
@noinline begin_for(result::T, start::I, stop::I) where {I, T} = new_intrinsic()::Tuple{Int, T}

@noinline yield_for(val::T=nothing) where T = new_intrinsic()::T

const BrutusType = Union{Bool,Int64,Int32,Float32,Float64,UInt64,Array{Float64},Array{Int64}}

function cmpi_pred(predicate)
    function(ops; loc=Location())
        arith.cmpi(predicate, ops; loc)
    end
end

function single_op_wrapper(fop)
    (block::Block, args::Vector{Value}, arg_types::Vector{<:Type}; loc=Location()) -> IR.get_result(push!(block, fop(args; loc)))
end

intrinsics_to_mlir = Dict([
    Base.and_int => single_op_wrapper(arith.andi),
    Base.add_int => single_op_wrapper(arith.addi),
    Base.sub_int => single_op_wrapper(arith.subi),
    Base.sle_int => single_op_wrapper(cmpi_pred(arith.Predicates.sle)),
    Base.slt_int => single_op_wrapper(cmpi_pred(arith.Predicates.slt)),
    Base.ult_int => single_op_wrapper(cmpi_pred(arith.Predicates.slt)),
    Base.:(===) =>  single_op_wrapper(cmpi_pred(arith.Predicates.eq)),
    Base.mul_int => single_op_wrapper(arith.muli),
    Base.mul_float => single_op_wrapper(arith.mulf),
    Base.add_float => single_op_wrapper(arith.addf),
    Base.not_int => function(block, args, arg_types; loc=Location())
        arg = only(args)
        ones = push!(block, arith.constant(-1, IR.get_type(arg); loc)) |> IR.get_result
        IR.get_result(push!(block, arith.xori(Value[arg, ones]; loc)))
    end,
    Base.bitcast => function(block, args, arg_types; loc=Location())
        type, value = args
        IR.get_result(push!(block, Arith.Bitcast(; location=loc, out_=type, in_=value)))
    end,
    # Base.arraylen => function(block, args, arg_types; loc=Location())
    #     source_ = only(args)
    #     N = ndims(only(arg_types))
    #     total = IR.get_result(push!(block, arith.constant(1, IR.IndexType())))
    #     for i in 0:N-1
    #         index_ = IR.get_result(push!(block, arith.constant(i, IR.IndexType())))
    #         dim = IR.get_result(push!(block, Memref.Dim(; location=loc, result_=IR.IndexType(), source_, index_)))
    #         total = IR.get_result(push!(block, Index.Mul(; location=loc, result_=IR.IndexType(), lhs_=total, rhs_=dim)))
    #     end
    #     IR.get_result(push!(block, Index.CastS(; location=loc, output_=MLIRType(Int), input_=total)))
    # end,
    # Base.arrayref => function(block, args, arg_types; loc=Location())
    #     @info "ARRAYREF" arg_types
    #     _, memref_, indices_... = args
    #     one_off = IR.get_result(push!(block, arith.constant(1, IR.IndexType(); loc)))
    #     T = eltype(arg_types[2])
    #     N = ndims(arg_types[2])
    #     for (i, index) in enumerate(indices_)
    #         index = IR.get_result(push!(block, Index.CastS(; location=loc, output_=IR.IndexType(), input_=index)))
    #         indices_[i] = IR.get_result(push!(block, Index.Sub(; location=loc, result_=IR.IndexType(), lhs_=index, rhs_=one_off)))
    #     end
    #     if N > 1 && length(indices_) == 1 # linear indexing needs to be converted to cartesian
    #         linear_index = only(indices_)
    #         indices_ = Value[]
    #         for i in 0:N-1
    #             index_ = IR.get_result(push!(block, arith.constant(i, IR.IndexType())))
    #             dim = IR.get_result(push!(block, Memref.Dim(; location=loc, result_=IR.IndexType(), source_=memref_, index_)))
    #             sub = IR.get_result(push!(block, Index.RemS(; location=loc, result_=IR.IndexType(), lhs_=linear_index, rhs_=dim)))
    #             linear_index = IR.get_result(push!(block, Index.Sub(; location=loc, result_=IR.IndexType(), lhs_=linear_index, rhs_=sub)))
    #             linear_index = IR.get_result(push!(block, Index.DivS(; location=loc, result_=IR.IndexType(), lhs_=linear_index, rhs_=dim)))
    #             push!(indices_, sub)
    #         end
    #     end
    #     IR.get_result(push!(block, Memref.Load(; location=loc, result_=MLIRType(T), memref_, indices_)))
    # end,
    # Base.arrayset => function(block, args, arg_types; loc=Location())
    #     _, memref_, value_, indices_... = args
    #     one_off = IR.get_result(push!(block, arith.constant(1, IR.IndexType(); loc)))
    #     N = ndims(arg_types[2])
    #     for (i, index) in enumerate(indices_)
    #         index = IR.get_result(push!(block, Index.CastS(; location=loc, output_=IR.IndexType(), input_=index)))
    #         indices_[i] = IR.get_result(push!(block, Index.Sub(; location=loc, result_=IR.IndexType(), lhs_=index, rhs_=one_off)))
    #     end
    #     if N > 1 && length(indices_) == 1 # linear indexing needs to be converted to cartesian
    #         linear_index = only(indices_)
    #         indices_ = Value[]
    #         for i in 0:N-1
    #             index_ = IR.get_result(push!(block, arith.constant(i, IR.IndexType())))
    #             dim = IR.get_result(push!(block, Memref.Dim(; location=loc, result_=IR.IndexType(), source_=memref_, index_)))
    #             sub = IR.get_result(push!(block, Index.RemS(; location=loc, result_=IR.IndexType(), lhs_=linear_index, rhs_=dim)))
    #             linear_index = IR.get_result(push!(block, Index.Sub(; location=loc, result_=IR.IndexType(), lhs_=linear_index, rhs_=sub)))
    #             linear_index = IR.get_result(push!(block, Index.DivS(; location=loc, result_=IR.IndexType(), lhs_=linear_index, rhs_=dim)))
    #             push!(indices_, sub)
    #         end
    #     end
    #     push!(block, Memref.Store(; location=loc, value_, memref_, indices_))
        
    #     memref_
    # end,
    Base.arraysize => function(block, args, arg_types; loc=Location())
        source_, index_ = args
        one_off = IR.get_result(push!(block, arith.constant(1, IR.IndexType(); loc)))
        index_ = IR.get_result(push!(block, Index.CastS(; location=loc, output_=IR.IndexType(), input_=index_)))
        index_ = IR.get_result(push!(block, Index.Sub(; location=loc, result_=IR.IndexType(), lhs_=index_, rhs_=one_off)))
        input_ = IR.get_result(push!(block, Memref.Dim(; location=loc, result_=IR.IndexType(), source_, index_)))
        IR.get_result(push!(block, Index.CastS(; location=loc, output_=MLIRType(Int), input_)))
    end,
    Core.ifelse => function(block, args, arg_types; loc=Location())
        @assert arg_types[2] == arg_types[3] "Branches in Core.ifelse should have the same type."
        condition_, true_value_, false_value_ = args
        IR.get_result(push!(block, Arith.Select(; location=loc, result_=MLIRType(arg_types[2]), condition_, true_value_, false_value_)))
    end,
])

"Generates a block argument for each phi node present in the block."
function prepare_block(ir, bb)
    b = Block()

    for sidx in bb.stmts
        stmt = ir.stmts[sidx]
        inst = stmt[:inst]
        inst isa Core.PhiNode || continue

        type = stmt[:type]
        IR.push_argument!(b, MLIRType(type), Location())
    end

    return b
end

"Values to populate the Phi Node when jumping from `from` to `to`."
function collect_value_arguments(ir, from, to)
    to = ir.cfg.blocks[to]
    values = []
    for s in to.stmts
        stmt = ir.stmts[s]
        inst = stmt[:inst]
        inst isa Core.PhiNode || continue

        edge = findfirst(==(from), inst.edges)
        if isnothing(edge) # use dummy scalar val instead
            val = zero(stmt[:type])
            push!(values, val)
        else
            push!(values, inst.values[edge])
        end
    end
    values
end

"""
    code_mlir(f, types::Type{Tuple}) -> IR.Operation

Returns a `func.func` operation corresponding to the ircode of the provided method.
This only supports a few Julia Core primitives and scalar types of type $BrutusType.

!!! note
    The Julia SSAIR to MLIR conversion implemented is very primitive and only supports a
    handful of primitives. A better to perform this conversion would to create a dialect
    representing Julia IR and progressively lower it to base MLIR dialects.
"""
function code_mlir(f, types)
    ctx = context()
    ir, ret = Core.Compiler.code_ircode(f, types) |> only
    @assert first(ir.argtypes) isa Core.Const

    values = Vector{Value}(undef, length(ir.stmts))

    for dialect in (true ? ("func", "cf") : ("std",))
        IR.get_or_load_dialect!(dialect)
    end

    blocks = [
        prepare_block(ir, bb)
        for bb in ir.cfg.blocks
    ]

    regions = [Region()]
    loop_results = []
    current_block = entry_block = blocks[begin]
    
    @info "Function arguments: $(types.parameters)"
    for argtype in types.parameters
        IR.push_argument!(entry_block, MLIRType(argtype), Location())
    end

    function get_value(x)::Union{Value, MLIRType}
        if x isa Core.SSAValue
            @assert isassigned(values, x.id) "value $x was not assigned"
            return values[x.id]
        elseif x isa Core.Argument
            return IR.get_argument(entry_block, x.n - 1)
        elseif x isa BrutusType
            return IR.get_result(push!(current_block, arith.constant(x)))
        elseif (x isa Type) && (x <: BrutusType)
            return IR.MLIRType(x)
        elseif x == GlobalRef(Main, :nothing) # This might be something else than Main sometimes?
            return IR.MLIRType(Nothing)
        else
            error("could not use value $x inside MLIR")
        end
    end

    function get_type(ir, x)::Union{DataType, Nothing}
        if x isa Core.SSAValue
            return ir.stmts.type[x.id]
        elseif x isa Core.Argument
            return ir.argtypes[x.n]
        elseif x isa BrutusType
            return typeof(x)
        else
            @info "Could not get type for $x, of type $(typeof(x))."
            return nothing
            # error("could not get type for $x, of type $(typeof(x))")
        end
    end

    for (block_id, (b, bb)) in enumerate(zip(blocks, ir.cfg.blocks))
        current_block = b
        @info "Pushing to region ($(length(regions)))."
        push!(regions[end], current_block)
        n_phi_nodes = 0

        for sidx in bb.stmts
            stmt = ir.stmts[sidx]
            inst = stmt[:inst]
            @info "Working on: $(inst)"
            if inst == nothing
                inst = Core.GotoNode(block_id+1)
                line = Core.LineInfoNode(Brutus, :code_mlir, Symbol(@__FILE__), Int32(@__LINE__), Int32(@__LINE__))
            else
                line = ir.linetable[stmt[:line]]
            end

            if Meta.isexpr(inst, :call)
                val_type = stmt[:type]
                if !(val_type <: BrutusType)
                    error("type $val_type is not supported")
                end
                out_type = MLIRType(val_type)
                
                called_func = first(inst.args)
                if called_func isa GlobalRef # TODO: should probably use something else here
                    called_func = getproperty(called_func.mod, called_func.name)
                end

                # temp = @view inst.args[begin+1:end]
                # args = map(temp) do arg
                #     @show arg
                #     arg = get_value(arg)
                #     arg
                # end
                args = get_value.(@view inst.args[begin+1:end])

                arg_types = get_type.(Ref(ir), @view inst.args[begin+1:end])
                
                if (called_func == Base.getfield)
                    op = Operation(API.mlirOpResultGetOwner(args[1]), owned=false)
                    values[sidx] = IR.get_result(op, inst.args[3])
                    @show IR.get_type(values[sidx])
                    
                else
                    fop! = intrinsics_to_mlir[called_func]

                    loc = Location(string(line.file), line.line, 0)
                    res = fop!(current_block, args, arg_types; loc)

                    values[sidx] = res
                end

            elseif Meta.isexpr(inst, :invoke)
                called_func = inst.args[begin+1]
                if called_func isa GlobalRef
                    called_func = getproperty(called_func.mod, called_func.name)
                end
                val_type = stmt[:type]

                loc = Location(string(line.file), line.line, 0)

                # entries 1 and 2 of these arrays contain :invoke and :(Main.Brutus.begin_for) respectively, these are unimportant
                args = get_value.(@view inst.args[begin+2:end])
                arg_types = get_type.(Ref(ir), @view inst.args[begin+2:end])

                if (called_func == Brutus.begin_for)
                    next_block = blocks[block_id+1]
                    
                    # the first input is always the loop index
                    inputs_ = Value[IR.push_argument!(next_block, IR.IndexType(), IR.Location())]

                    outputs_ = MLIRType[IR.IndexType()]

                    if (stmt[:type] <: Tuple) # this signifies an additional loop variable
                        loopvartype = MLIRType(fieldtypes(stmt[:type])[2]) # for now, only one loop variable possible
                        push!(outputs_, loopvartype)
                        push!(inputs_, IR.push_argument!(next_block, loopvartype, loc))
                    end

                    value = push!(next_block, Builtin.UnrealizedConversionCast(;
                        location=loc,
                        outputs_,
                        inputs_
                    )) |> IR.get_result # only store the first result. if later, the second is needed it can be recovered still.
                    
                    # return regions

                    @warn value

                    values[sidx] = value
                    loop_region = Region()
                    push!(regions, loop_region)
                
                    initial_values_..., start_, stop_ = args
                    initial_values_types..., _, _ = arg_types

                    for_op = push!(current_block, Affine.For(;
                        location=loc,
                        results_=MLIRType.(initial_values_types),
                        start_,
                        stop_,
                        initial_values_,
                        region_=loop_region
                    ))
                    for_result = IR.num_results(for_op) > 0 ? IR.get_result(for_op) : nothing
                    push!(loop_results, for_result)
                    current_block = next_block # a hack to make sure that getfields are added inside the loop body
                elseif (called_func == Brutus.yield_for)
                    args = Value[args...]
                    push!(current_block, Affine.Yield(;
                        location=loc,
                        operands_=args
                    ))
                    @assert !regions[end].owned
                    pop!(regions)
                    loop_result = pop!(loop_results)
                    if loop_result != nothing; values[sidx] = loop_result; end

                else
                    if !(val_type <: BrutusType)
                        error("type $val_type is not supported")
                    end
                    out_type = MLIRType(val_type)
                    
                    fop! = intrinsics_to_mlir[called_func]

                    res = IR.get_result(fop!(current_block, args, arg_types; loc))

                    values[sidx] = res
                end

            elseif inst isa PhiNode
                values[sidx] = IR.get_argument(current_block, n_phi_nodes += 1)
            elseif inst isa PiNode
                values[sidx] = get_value(inst.val)
            elseif inst isa GotoNode
                args = get_value.(collect_value_arguments(ir, block_id, inst.label))
                dest = blocks[inst.label]
                loc = Location(string(line.file), line.line, 0)
                brop = true ? cf.br : std.br
                push!(current_block, brop(dest, args; loc))
            elseif inst isa GotoIfNot
                false_args = get_value.(collect_value_arguments(ir, block_id, inst.dest))
                cond = get_value(inst.cond)
                @assert length(bb.succs) == 2 # NOTE: We assume that length(bb.succs) == 2, this might be wrong
                other_dest = setdiff(bb.succs, inst.dest) |> only
                true_args = get_value.(collect_value_arguments(ir, block_id, other_dest))
                other_dest = blocks[other_dest]
                dest = blocks[inst.dest]

                loc = Location(string(line.file), line.line, 0)
                cond_brop = true ? cf.cond_br : std.cond_br
                cond_br = cond_brop(cond, other_dest, dest, true_args, false_args; loc)
                push!(current_block, cond_br)
            elseif inst isa ReturnNode
                line = ir.linetable[stmt[:line]]
                retop = true ? func.return_ : std.return_
                loc = Location(string(line.file), line.line, 0)
                push!(current_block, retop([get_value(inst.val)]; loc))
            elseif Meta.isexpr(inst, :code_coverage_effect)
                # Skip
            else
                error("unhandled ir $(inst)")
            end
        end
    end

    func_name = nameof(f)

    LLVM15 = true

    input_types = MLIRType[
        IR.get_type(IR.get_argument(entry_block, i))
        for i in 1:IR.num_arguments(entry_block)
    ]
    result_types = [MLIRType(ret)]

    ftype = MLIRType(input_types => result_types)
    op = IR.create_operation(
        LLVM15 ? "func.func" : "builtin.func",
        Location();
        attributes = [
            NamedAttribute("sym_name", IR.Attribute(string(func_name))),
            NamedAttribute(LLVM15 ? "function_type" : "type", IR.Attribute(ftype)),
            NamedAttribute("llvm.emit_c_interface", IR.Attribute(API.mlirUnitAttrGet(IR.context())))
        ],
        owned_regions = Region[regions[end]],
        result_inference=false,
    )

    IR.verifyall(op)

    op
end

"""
    @code_mlir f(args...)
"""
macro code_mlir(call)
    @assert Meta.isexpr(call, :call) "only calls are supported"

    f = first(call.args) |> esc
    args = Expr(:curly,
        Tuple,
        map(arg -> :($(Core.Typeof)($arg)),
            call.args[begin+1:end])...,
    ) |> esc

    quote
        code_mlir($f, $args)
    end
end

end # module Brutus
