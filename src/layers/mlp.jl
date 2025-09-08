"""
    MLP(in_dims::Integer, hidden_dims::Dims{N}, activation=NNlib.relu; norm_layer=nothing,
        dropout_rate::Real=0.0f0, dense_kwargs=(;), norm_kwargs=(;),
        last_layer_activation=false) where {N}

Construct a multi-layer perceptron (MLP) with dense layers, optional normalization layers,
and dropout.

## Arguments

  - `in_dims`: number of input dimensions
  - `hidden_dims`: dimensions of the hidden layers
  - `activation`: activation function (stacked after the normalization layer, if present
    else after the dense layer)

## Keyword Arguments

  - `norm_layer`: $(NORM_LAYER_DOC)
  - `dropout_rate`: dropout rate (default: `0.0f0`)
  - `dense_kwargs`: keyword arguments for the dense layers
  - `norm_kwargs`: keyword arguments for the normalization layers
  - `last_layer_activation`: set to `true` to apply the activation function to the last
    layer
  - `residual_connection`: set to `true` to apply a residual connection to the MLP
"""
@concrete struct MLP <: AbstractLuxWrapperLayer{:chain}
    chain <: Union{Lux.Chain,Lux.SkipConnection}
end

function MLP(in_dims::Integer, hidden_dims::Vector{<:Integer}, args...; kwargs...)
    return MLP(in_dims, Dims(hidden_dims), args...; kwargs...)
end

function MLP(
    in_dims::Integer,
    hidden_dims::Dims{N},
    activation::F=NNlib.relu;
    norm_layer::NF=nothing,
    dropout_rate::Real=0.0f0,
    last_layer_activation::Bool=false,
    dense_kwargs=(;),
    norm_kwargs=(;),
    residual_connection::Bool=false,
) where {N,F,NF}
    @argcheck N > 0
    layers = Vector{AbstractLuxLayer}(undef, N)
    for (i, out_dims) in enumerate(hidden_dims)
        act = i != N ? activation : (last_layer_activation ? activation : identity)
        layers[i] = dense_norm_act_dropout(
            i, in_dims => out_dims, act, norm_layer, dropout_rate, dense_kwargs, norm_kwargs
        )
        if residual_connection && in_dims == out_dims
            layers[i] = Lux.SkipConnection(layers[i], +)
        end
        in_dims = out_dims
    end
    inner_blocks = NamedTuple{ntuple(i -> Symbol(:block, i), N)}(layers)
    return MLP(Lux.Chain(inner_blocks))
end

function dense_norm_act_dropout(
    i::Integer,
    (in_dims, out_dims)::Pair{<:Integer,<:Integer},
    activation::F,
    norm_layer::NF,
    dropout_rate::Real,
    dense_kwargs,
    norm_kwargs,
) where {F,NF}
    if iszero(dropout_rate)
        if norm_layer === nothing
            return Lux.Dense(in_dims => out_dims, activation; dense_kwargs...)
        end
        return Lux.Chain(;
            dense=Lux.Dense(in_dims => out_dims; dense_kwargs...),
            norm=norm_layer(i, out_dims, activation; norm_kwargs...),
        )
    end
    if norm_layer === nothing
        return Lux.Chain(;
            dense=Lux.Dense(in_dims => out_dims, activation; dense_kwargs...),
            dropout=Lux.Dropout(dropout_rate),
        )
    end
    return Lux.Chain(;
        dense=Lux.Dense(in_dims => out_dims; dense_kwargs...),
        norm=norm_layer(i, out_dims, activation; norm_kwargs...),
        dropout=Lux.Dropout(dropout_rate),
    )
end
