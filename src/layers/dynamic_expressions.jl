"""
    DynamicExpressionsLayer(operator_enum::OperatorEnum, expressions::Node...;
        eval_options::EvalOptions=EvalOptions())
    DynamicExpressionsLayer(operator_enum::OperatorEnum,
        expressions::AbstractVector{<:Node}; kwargs...)

Wraps a `DynamicExpressions.jl` `Node` into a Lux layer and allows the constant nodes to
be updated using any of the AD Backends.

For details about these expressions, refer to the
[`DynamicExpressions.jl` documentation](https://symbolicml.org/DynamicExpressions.jl/dev/types/).

## Arguments

  - `operator_enum`: `OperatorEnum` from `DynamicExpressions.jl`
  - `expressions`: `Node` from `DynamicExpressions.jl` or `AbstractVector{<:Node}`

## Keyword Arguments

  - `turbo`: Use `LoopVectorization.jl` for faster evaluation **(Deprecated)**
  - `bumper`: Use `Bumper.jl` for faster evaluation **(Deprecated)**
  - `eval_options`: EvalOptions from `DynamicExpressions.jl`

These options are simply forwarded to `DynamicExpressions.jl`'s `eval_tree_array`
and `eval_grad_tree_array` function.

# Extended Help

## Example

```jldoctest
julia> operators = OperatorEnum(; binary_operators=[+, -, *], unary_operators=[cos]);

julia> x1 = Node(; feature=1);

julia> x2 = Node(; feature=2);

julia> expr_1 = x1 * cos(x2 - 3.2)
x1 * cos(x2 - 3.2)

julia> expr_2 = x2 - x1 * x2 + 2.5 - 1.0 * x1
((x2 - (x1 * x2)) + 2.5) - (1.0 * x1)

julia> layer = Layers.DynamicExpressionsLayer(operators, expr_1, expr_2);

julia> ps, st = Lux.setup(Random.default_rng(), layer)
((layer_1 = (layer_1 = (params = Float32[3.2],), layer_2 = (params = Float32[2.5, 1.0],)), layer_2 = NamedTuple()), (layer_1 = (layer_1 = NamedTuple(), layer_2 = NamedTuple()), layer_2 = NamedTuple()))

julia> x = [1.0f0 2.0f0 3.0f0
            4.0f0 5.0f0 6.0f0]
2×3 Matrix{Float32}:
 1.0  2.0  3.0
 4.0  5.0  6.0

julia> layer(x, ps, st)[1] ≈ Float32[0.6967068 -0.4544041 -2.8266668; 1.5 -4.5 -12.5]
true

julia> ∂x, ∂ps, _ = Zygote.gradient(Base.Fix1(sum, abs2) ∘ first ∘ layer, x, ps, st);

julia> ∂x ≈ Float32[-14.0292 54.206482 180.32669; -0.9995737 10.7700815 55.6814]
true

julia> ∂ps.layer_1.layer_1.params ≈ Float32[-6.451908]
true

julia> ∂ps.layer_1.layer_2.params ≈ Float32[-31.0, 90.0]
true
```
"""
@concrete struct DynamicExpressionsLayer <: AbstractLuxWrapperLayer{:chain}
    chain
end

@concrete struct InternalDynamicExpressionWrapper <: AbstractLuxLayer
    operator_enum
    expression
    eval_options
end

function Base.show(io::IO, l::InternalDynamicExpressionWrapper)
    print(io,
        "InternalDynamicExpressionWrapper($(l.operator_enum), $(l.expression); \
         eval_options=$(l.eval_options))")
end

function LuxCore.initialparameters(::AbstractRNG, layer::InternalDynamicExpressionWrapper)
    params = map(Base.Fix2(getproperty, :val),
        filter(
            node -> node.degree == 0 && node.constant,
            dynamic_expression_get_node(layer.expression)
        )
    )
    return (; params)
end

function update_de_expression_constants!(expression, ps)
    # Don't use `set_constant_refs!` here, since it requires the types to match. In our
    # case we just warn the user
    params = filter(
        node -> node.degree == 0 && node.constant,
        dynamic_expression_get_node(expression)
    )
    foreach(enumerate(params)) do (i, node)
        (node.val isa typeof(ps[i])) ||
            @warn lazy"node.val::$(typeof(node.val)) != ps[$i]::$(typeof(ps[i])). Type of node.val takes precedence. Fix the input expression if this is unintended." maxlog=1
        return node.val = ps[i]
    end
    return
end

function (de::InternalDynamicExpressionWrapper)(x::AbstractVector, ps, st)
    y, stₙ = de(reshape(x, :, 1), ps, st)
    return vec(y), stₙ
end

# NOTE: Can't use `get_device_type` since it causes problems with ReverseDiff
function (de::InternalDynamicExpressionWrapper)(x::AbstractMatrix, ps, st)
    y = apply_dynamic_expression(de, de.expression, de.operator_enum,
        Lux.match_eltype(de, ps, st, x), ps.params, get_device(x))
    return y, st
end

function apply_dynamic_expression(
        de::InternalDynamicExpressionWrapper, expr, operator_enum, x, ps, ::CPUDevice)
    if !is_extension_loaded(Val(:DynamicExpressions))
        error("`DynamicExpressions.jl` is not loaded. Please load it before using \
               `DynamicExpressionsLayer`.")
    end
    return apply_dynamic_expression_internal(de, expr, operator_enum, x, ps)
end

function apply_dynamic_expression(de, expr, operator_enum, x, ps, dev)
    throw(ArgumentError("`DynamicExpressions.jl` only supports CPU operations. Current \
                         device detected as $(dev). CUDA.jl will be supported after \
                         https://github.com/SymbolicML/DynamicExpressions.jl/pull/65 is \
                         merged upstream."))
end

function CRC.rrule(
        ::typeof(apply_dynamic_expression), de::InternalDynamicExpressionWrapper,
        expr, operator_enum, x, ps, ::CPUDevice)
    if !is_extension_loaded(Val(:DynamicExpressions))
        error("`DynamicExpressions.jl` is not loaded. Please load it before using \
               `DynamicExpressionsLayer`.")
    end
    return ∇apply_dynamic_expression(de, expr, operator_enum, x, ps)
end

function apply_dynamic_expression_internal end

function ∇apply_dynamic_expression end

function dynamic_expression_get_node end
