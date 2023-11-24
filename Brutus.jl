module Brutus

__revise_mode__ = :eval

import LLVM
using MLIR.IR
using MLIR: API
using MLIR.Dialects: arith, func, cf, std, Arith, Memref, Index, Builtin, Ub, Affine, Llvm
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode

struct MemRef{T,N} <: DenseArray{T, N}
    allocated_pointer::Ptr{T}
    aligned_pointer::Ptr{T}
    offset::Int
    sizes::NTuple{N, Int}
    strides::NTuple{N, Int}
    data::Array{T, N}
end
import Base.show
Base.show(io::IO, A::Brutus.MemRef{T, N}) where {T, N} = print(io, "Brutus.MemRef{$T,$N} (size $(join(A.sizes, "×")))")
Base.show(io::IO, ::MIME{Symbol("text/plain")}, X::Brutus.MemRef) = show(io, X)

function MemRef(a::Array{T,N}) where {T,N}
    @assert isbitstype(T) "Array element type should be isbits, got $T."
    allocated_pointer = a.ref.mem.ptr
    aligned_pointer = a.ref.ptr_or_offset
    offset = Int((aligned_pointer - allocated_pointer)//sizeof(T))
    @assert offset == 0 "Arrays with Memoryref offset are, as of yet, unsupported."
    sizes = size(a)
    strides = Tuple([1, cumprod(sizes)[1:end-1]...])

    return MemRef{T,N}(
        allocated_pointer,
        aligned_pointer,
        offset,
        sizes,
        strides,
        a,
    )
end



new_intrinsic = ()->Base.compilerbarrier(:const, error("Intrinsics should be compiled to MLIR!"))

@noinline begin_for(start::I, stop::I) where {I <: Integer} =  new_intrinsic()::Int
@noinline begin_for(result::T, start::I, stop::I) where {I, T} = new_intrinsic()::Tuple{Int, T}
@noinline mlir_load(A::Brutus.MemRef{T}, I::Union{Integer, CartesianIndex}...) where T = Brutus.new_intrinsic()::T
@noinline mlir_store!(A::Brutus.MemRef{T}, v, I::Union{Integer, CartesianIndex}...) where {T} = new_intrinsic()::T

@noinline delinearize_index(i1::Integer, basis::NTuple{N, Int}) where {N} = Brutus.new_intrinsic()::NTuple{N, Int}
delinearize_index(i1::Integer, A::Brutus.MemRef) = delinearize_index(i1, size(A))

@noinline yield_for(val::T=nothing) where T = new_intrinsic()::T

IR.MLIRType(::Type{Nothing}) = IR.MLIRType(API.mlirLLVMVoidTypeGet(IR.context()))

const BrutusScalarType = Union{Bool, Int64, UInt64, Int32, UInt32, Float32, Float64, UInt64, Array{Float64}, Array{Int64}}
const BrutusType = Union{BrutusScalarType, Array{BrutusScalarType}}

mutable struct CodegenContext
    regions::Vector{Region}
    loop_thunks::Vector
    const blocks::Vector{Block}
    const entryblock::Block
    currentblockindex::Int
    const ir::Core.Compiler.IRCode
    const ret::Type
    const values::Vector
    const args::Vector
end
currentblock(cg::CodegenContext) = cg.blocks[cg.currentblockindex]
start_region!(cg::CodegenContext) = push!(cg.regions, Region())
stop_region!(cg::CodegenContext) = pop!(cg.regions)
currentregion(cg::CodegenContext) = cg.regions[end]

function get_value(cg::CodegenContext, x)
    if x isa Core.SSAValue
        @assert isassigned(cg.values, x.id) "value $x was not assigned"
        return cg.values[x.id]
    elseif x isa Core.Argument
        @assert isassigned(cg.args, x.n-1) "value $x was not assigned"
        return cg.args[x.n-1]
        # return IR.get_argument(cg.entryblock, x.n - 1)
    elseif x isa BrutusType
        return IR.get_result(push!(currentblock(cg), arith.constant(x)))
    elseif (x isa Type) && (x <: BrutusType)
        return IR.MLIRType(x)
    elseif x == GlobalRef(Main, :nothing) # This might be something else than Main sometimes?
        return IR.MLIRType(Nothing)
    else
        error("could not use value $x inside MLIR")
    end
end

function get_type(cg::CodegenContext, x)
    if x isa Core.SSAValue
        return cg.ir.stmts.type[x.id]
    elseif x isa Core.Argument
        return cg.ir.argtypes[x.n]
    elseif x isa BrutusType
        return typeof(x)
    else
        @warn "Could not get type for $x, of type $(typeof(x))."
        return nothing
        # error("could not get type for $x, of type $(typeof(x))")
    end
end

struct InstructionContext
    args::Vector
    result_type::Type
    loc::Location
end

function cmpi_pred(predicate)
    function(ops; loc=Location())
        arith.cmpi(predicate, ops; loc)
    end
end

function single_op_wrapper(fop)
    (cg::CodegenContext, ic::InstructionContext)->IR.get_result(push!(currentblock(cg), fop(indextoi64.(Ref(cg), get_value.(Ref(cg), ic.args)))))
end

indextoi64(cg::CodegenContext, x; loc=IR.Location()) = x
function indextoi64(cg::CodegenContext, x::Value; loc=IR.Location())
    mlirtype = IR.get_type(x)
    if API.mlirTypeIsAIndex(mlirtype)
        return push!(currentblock(cg), Index.CastS(;
            location=loc,
            output_=MLIRType(Int),
            input_=x
        )) |> IR.get_result
    else
        return x
    end
end
function i64toindex(cg, x::Value; loc=IR.Location())
    mlirtype = IR.get_type(x)
    if API.mlirTypeIsAInteger(mlirtype)
        return push!(currentblock(cg), Index.CastS(;
            location=loc,
            output_=IR.IndexType(),
            input_=x
        )) |> IR.get_result
    else
        return x
    end
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
    Base.not_int => function(cg::CodegenContext, ic::InstructionContext)
        arg = get_value(cg, only(ic.args))
        ones = push!(currentblock(cg), arith.constant(-1, IR.get_type(arg); ic.loc)) |> IR.get_result
        IR.get_result(push!(currentblock(cg), arith.xori(Value[arg, ones]; ic.loc)))
    end,
    Base.bitcast => function(cg::CodegenContext, ic::InstructionContext)
        @show ic.args
        type, value = get_value.(Ref(cg), ic.args)
        value = indextoi64(cg, value)
        IR.get_result(push!(currentblock(cg), Arith.Bitcast(; location=ic.loc, out_=type, in_=value)))
    end,
    Base.getfield => function(cg::CodegenContext, ic::InstructionContext)
        object = get_value(cg, first(ic.args))
        field = ic.args[2]
        if field isa QuoteNode; field=field.value; end
        return getfield(object, field)
    end,
    Core.tuple => function(cg::CodegenContext, ic::InstructionContext)
        inputs_ = get_value.(Ref(cg), ic.args)
        outputs_ = MLIRType.(get_type.(Ref(cg), ic.args))
        
        op = push!(currentblock(cg), Builtin.UnrealizedConversionCast(;
            location=ic.loc,
            outputs_,
            inputs_
        ))
        return Tuple(IR.get_result.(Ref(op), 1:fieldcount(ic.result_type)))
    end,
    Core.ifelse => function(block, args, arg_types; loc=Location(), ir)
        @assert arg_types[2] == arg_types[3] "Branches in Core.ifelse should have the same type."
        condition_, true_value_, false_value_ = args
        IR.get_result(push!(block, Arith.Select(; location=loc, result_=MLIRType(arg_types[2]), condition_, true_value_, false_value_)))
    end,
    Base.throw_boundserror => function(cg::CodegenContext, ic::InstructionContext)
        @warn "Ignoring potential boundserror while generating MLIR."
        return nothing
    end,
    Core.memoryref => function(cg::CodegenContext, ic::InstructionContext)
        @assert get_type(cg, ic.args[1]) <: MemoryRef "memoryref(::Memory) is not yet supported."
        mr = get_value(cg, ic.args[1])
        one_off = IR.get_result(push!(currentblock(cg), arith.constant(1, IR.IndexType(); ic.loc)))
        offsets_ = push!(currentblock(cg), Index.Sub(;
            location=ic.loc,
            result_=IR.IndexType(),
            lhs_=i64toindex(cg, get_value(cg, ic.args[2])),
            rhs_=one_off
        )) |> IR.get_results
        sizes_ = push!(currentblock(cg), Index.Sub(;
            location=ic.loc,
            result_=IR.IndexType(),
            lhs_=mr.mem.length,
            rhs_=only(offsets_)
        )) |> IR.get_results
        flattened = push!(currentblock(cg), Memref.ReinterpretCast(;
            location=Location(),
            result_=MLIRType(Vector{eltype(get_type(cg, ic.args[1]))}),
            source_=mr.ptr_or_offset,
            offsets_,
            sizes_,
            strides_=Value[],
            static_offsets_=IR.Attribute(API.mlirDenseI64ArrayGet(context().context, 1, Int[typemin(Int64)])),
            static_sizes_=IR.Attribute(API.mlirDenseI64ArrayGet(context().context, 1, Int[typemin(Int64)])),
            static_strides_=IR.Attribute(API.mlirDenseI64ArrayGet(context().context, 1, Int[1]))
        )) |> IR.get_result
        return (; ptr_or_offset=flattened, mem=mr.mem)
    end,
    Core.memoryrefget => function(cg::CodegenContext, ic::InstructionContext)
        @assert ic.args[2] == :not_atomic "Only non-atomic memoryrefget is supported."
        # TODO: ic.args[3] signals boundschecking, currently ignored.
        
        memref_ = get_value(cg, ic.args[1]).ptr_or_offset
        indices_=push!(currentblock(cg), arith.constant(0, IR.IndexType(); ic.loc)) |> IR.get_results
        push!(currentblock(cg), Memref.Load(;
            location=ic.loc,
            result_=MLIRType(eltype(get_type(cg, ic.args[1]))),
            memref_,
            indices_
        )) |> IR.get_result
    end,
    Core.memoryrefset! => function(cg::CodegenContext, ic::InstructionContext)
        @assert ic.args[3] == :not_atomic "Only non-atomic memoryrefset! is supported."

        mr = get_value(cg, ic.args[1])

        value_ = get_value(cg, ic.args[2])
        memref_ = mr.ptr_or_offset
        indices_=push!(currentblock(cg), arith.constant(0, IR.IndexType(); ic.loc)) |> IR.get_results
        push!(currentblock(cg), Memref.Store(;
            location=ic.loc,
            value_,
            memref_,
            indices_
        ))
        return mr
    end,
    Brutus.begin_for => function(cg::CodegenContext, ic::InstructionContext)
        @assert length(cg.blocks) >= cg.currentblockindex + 1 "Not enough blocks in the CodegenContext."
        next_block = cg.blocks[cg.currentblockindex + 1]
                        
        # the first input is always the loop index
        value = IR.push_argument!(next_block, IR.IndexType(), ic.loc)

        if (ic.result_type <: Tuple) # this signifies an additional loop variable
            loopvartype = MLIRType(fieldtypes(ic.result_type)[2]) # for now, only one loop variable possible
            value = (value, IR.push_argument!(next_block, loopvartype, ic.loc))
        end

        start_region!(cg)

        initial_values_..., start_, stop_ = get_value.(Ref(cg), ic.args)
        initial_values_types..., _, _ = get_type.(Ref(cg), ic.args)

        start_, stop_ = i64toindex.(Ref(cg), (start_, stop_))
        stop_ = push!(currentblock(cg), Index.Add(;
            location = ic.loc,
            result_=IR.IndexType(),
            lhs_=stop_,
            rhs_=IR.get_result(push!(currentblock(cg), arith.constant(1, IR.IndexType(); ic.loc)))
        )) |> IR.get_result

        # affine.for operation can only be created when the region is complete.
        # Delaying the creation is OK since no other operations will be added to the current block after the for loop.
        cur_region = currentregion(cg)
        cur_block = currentblock(cg)
        cur_loc = ic.loc
        # TODO: here I assume the loop body only contains one block (as it should) this should be asserted.
        target_block = cg.blocks[cg.currentblockindex + 2]
        function for_thunk()
            for_op = push!(cur_block, Affine.For(;
                location=cur_loc,
                results_=MLIRType.(initial_values_types),
                start_,
                stop_,
                initial_values_,
                region_=cur_region
            ))
            push!(cur_block, cf.br(target_block, Value[]; loc=cur_loc))
            return IR.num_results(for_op) > 0 ? IR.get_result(for_op) : nothing
        end
        push!(cg.loop_thunks, for_thunk)
        cg.currentblockindex += 1 # a hack to make sure that getfields are added inside the loop body

        return value
    end,
    Brutus.yield_for => function (cg::CodegenContext, ic::InstructionContext)
        @assert length(cg.loop_thunks) > 0 "No loop to yield."
        push!(currentblock(cg), Affine.Yield(;
            location=ic.loc,
            operands_=Value[get_value.(Ref(cg), ic.args)...]
        ))
        for_thunk = pop!(cg.loop_thunks)
        for_result = for_thunk()
        stop_region!(cg)
        return for_result
    end,
    Brutus.delinearize_index => function (cg::CodegenContext, ic::InstructionContext)
        @show last(ic.args)
        linear_index_, basis_ = get_value.(Ref(cg), ic.args)
        rank = fieldcount(get_type(cg, last(ic.args)))
        linear_index_ = i64toindex(cg, linear_index_)
        return push!(currentblock(cg), Affine.DelinearizeIndex(;
            location=ic.loc,
            multi_index_=([IR.IndexType() for _ in 1:rank]),
            linear_index_,
            basis_=Value[basis_...]
        )) |> IR.get_results |> Tuple
    end,
    Brutus.mlir_load => function (cg::CodegenContext, ic::InstructionContext)
        mr, I... = get_value.(Ref(cg), ic.args)
        indices_ = i64toindex.(Ref(cg), I)
        return push!(currentblock(cg), Memref.Load(;
            location=ic.loc,
            result_=MLIRType(eltype(get_type(cg, ic.args[1]))),
            memref_=mr.aligned_pointer,
            indices_
        )) |> IR.get_result
    end,
    Brutus.mlir_store! => function (cg::CodegenContext, ic::InstructionContext)
        mr, value_, I... = get_value.(Ref(cg), ic.args)
        indices_ = i64toindex.(Ref(cg), I)
        push!(currentblock(cg), Memref.Store(;
            location=ic.loc,
            value_,
            memref_=mr.aligned_pointer,
            indices_
        ))
        return value_
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

    values = Vector(undef, length(ir.stmts))
    args = Vector(undef, length(types.parameters))
    for dialect in (true ? ("func", "cf") : ("std",))
        IR.get_or_load_dialect!(dialect)
    end

    blocks = [
        prepare_block(ir, bb)
        for bb in ir.cfg.blocks
    ]

    cg = CodegenContext(
        [Region()],
        [],
        blocks,
        blocks[begin],
        1,
        ir,
        ret,
        values,
        args
    )

    for (i, argtype) in enumerate(types.parameters)
        arg = IR.push_argument!(cg.entryblock, MLIRType(argtype), Location())

        if argtype <: DenseArray
            N = ndims(argtype)
            length = IR.get_result(push!(cg.entryblock, arith.constant(1, IR.IndexType())))
            sizes = []
            for i in 0:N-1
                index_ = IR.get_result(push!(cg.entryblock, arith.constant(i, IR.IndexType())))
                dim = IR.get_result(push!(cg.entryblock, Memref.Dim(; location=IR.Location(), result_=IR.IndexType(), source_=arg, index_)))
                push!(sizes, dim)
                length = IR.get_result(push!(cg.entryblock, Index.Mul(; location=IR.Location(), result_=IR.IndexType(), lhs_=length, rhs_=dim)))
            end
            sizes = Tuple(sizes)

            if argtype <: MemRef
                arg = (;
                    # allocated_pointer, # shouldn't occur in ircode so doesn't matter
                    aligned_pointer=arg,
                    # offset, # shouldn't occur in ircode so doesn't matter
                    sizes = sizes,
                )
            elseif argtype <: Array
                # This NamedTuple mirrors the layout of a Julia Array:
                arg = (;
                    ref=(;
                        ptr_or_offset=arg,
                        mem=(; length=length, ptr=nothing)
                    ),
                    size=sizes
                )
            end
            else throw("Array type $argtype not supported.")
        end

        println("adding argument $i")
        cg.args[i] = arg # Note that Core.Argument(index) ends up at index-1 in this array. We handle this in get_value.
        println(argtype, MLIRType(argtype))
    end

    for (block_id, bb) in enumerate(cg.ir.cfg.blocks)
        cg.currentblockindex = block_id
        @info "number of regions: $(length(cg.regions))"
        push!(currentregion(cg), currentblock(cg))
        n_phi_nodes = 0

        for sidx in bb.stmts
            stmt = cg.ir.stmts[sidx]
            inst = stmt[:inst]
            @info "Working on: $(inst)"
            if inst == nothing
                inst = Core.GotoNode(block_id+1)
                line = Core.LineInfoNode(Brutus, :code_mlir, Symbol(@__FILE__), Int32(@__LINE__), Int32(@__LINE__))
            else
                line = cg.ir.linetable[stmt[:line]]
            end

            if Meta.isexpr(inst, :call) || Meta.isexpr(inst, :invoke)
                val_type = stmt[:type]
                if Meta.isexpr(inst, :call)
                    called_func, args... = inst.args
                else # Meta.isexpr(inst, :invoke)
                    _, called_func, args... = inst.args
                end

                if called_func isa GlobalRef # TODO: should probably use something else here
                    called_func = getproperty(called_func.mod, called_func.name)
                end
                args = map(args) do arg
                    if arg isa GlobalRef
                        arg = getproperty(arg.mod, arg.name)
                    elseif arg isa QuoteNode
                        arg = arg.value
                    end
                    return arg
                end

                loc = Location(string(line.file), line.line, 0)
                ic = InstructionContext(args, val_type, loc)

                fop! = intrinsics_to_mlir[called_func]
                res = fop!(cg, ic)

                values[sidx] = res
            elseif inst isa PhiNode
                values[sidx] = IR.get_argument(currentblock(cg), n_phi_nodes += 1)
            elseif inst isa PiNode
                values[sidx] = get_value(values, inst.val)
            elseif inst isa GotoNode
                args = get_value.(Ref(cg), collect_value_arguments(cg.ir, cg.currentblockindex, inst.label))
                dest = cg.blocks[inst.label]
                loc = Location(string(line.file), line.line, 0)
                brop = true ? cf.br : std.br
                push!(currentblock(cg), brop(dest, args; loc))
            elseif inst isa GotoIfNot
                false_args = get_value.(Ref(cg), collect_value_arguments(cg.ir, cg.currentblockindex, inst.dest))
                cond = get_value(cg, inst.cond)
                @assert length(bb.succs) == 2 # NOTE: We assume that length(bb.succs) == 2, this might be wrong
                other_dest = setdiff(bb.succs, inst.dest) |> only
                true_args = get_value.(Ref(cg), collect_value_arguments(cg.ir, cg.currentblockindex, other_dest))
                other_dest = cg.blocks[other_dest]
                dest = cg.blocks[inst.dest]

                loc = Location(string(line.file), line.line, 0)
                cond_brop = true ? cf.cond_br : std.cond_br
                cond_br = cond_brop(cond, other_dest, dest, true_args, false_args; loc)
                push!(currentblock(cg), cond_br)
            elseif inst isa ReturnNode
                line = cg.ir.linetable[stmt[:line]]
                retop = true ? func.return_ : std.return_
                loc = Location(string(line.file), line.line, 0)

                returnvalue = isdefined(inst, :val) ? indextoi64(cg, get_value(cg, inst.val)) : IR.get_result(push!(currentblock(cg), Llvm.Undef(; location=loc, res_=MLIRType(cg.ret))))
                push!(currentblock(cg), retop([returnvalue]; loc))

            elseif Meta.isexpr(inst, :code_coverage_effect)
                # Skip
            elseif Meta.isexpr(inst, :boundscheck)
                @warn "discarding boundscheck"
                cg.values[sidx] = IR.get_result(push!(currentblock(cg), arith.constant(true)))
            else
                # @warn "unhandled ir $(inst)"
                # return inst
                error("unhandled ir $(inst)")
            end
        end        
    end
    
    func_name = nameof(f)
    
    # add fallthrough to next block if necessary
    for (i, b) in enumerate(cg.blocks)
        @show IR.mlirIsNull(API.mlirBlockGetTerminator(b))
        if (i != length(cg.blocks) && IR.mlirIsNull(API.mlirBlockGetTerminator(b)))
            @warn "Block $i did not have a terminator, adding one."
            args = []
            dest = cg.blocks[i+1]
            loc = IR.Location()
            brop = true ? cf.br : std.br
            push!(b, brop(dest, args; loc))
        end
    end

    LLVM15 = true

    input_types = MLIRType[
        IR.get_type(IR.get_argument(cg.entryblock, i))
        for i in 1:IR.num_arguments(cg.entryblock)
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
        owned_regions = Region[currentregion(cg)],
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
