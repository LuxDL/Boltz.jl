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
"""
function MLP(in_dims::Integer, hidden_dims::Dims{N}, activation::F=NNlib.relu;
        norm_layer::NF=nothing, dropout_rate::Real=0.0f0, last_layer_activation::Bool=false,
        dense_kwargs=(;), norm_kwargs=(;)) where {N, F, NF}
    @argcheck N > 0
    name = "MLP(in_dims=$in_dims, hidden_dims=$(hidden_dims[1:(N - 1)]), \
            out_dims=$(hidden_dims[N]))"
    layers = Vector{AbstractExplicitLayer}(undef, N)
    for (i, out_dims) in enumerate(hidden_dims)
        act = i != N ? activation : (last_layer_activation ? activation : identity)
        layers[i] = dense_norm_act_dropout(i, in_dims => out_dims, act, norm_layer,
            dropout_rate, dense_kwargs, norm_kwargs)
        in_dims = out_dims
    end
    inner_blocks = NamedTuple{ntuple(i -> Symbol(:block, i), N)}(layers)
    return Lux.Chain(inner_blocks; name, disable_optimizations=true)
end

function dense_norm_act_dropout(
        i::Integer, (in_dims, out_dims)::Pair{<:Integer, <:Integer}, activation::F,
        norm_layer::NF, dropout_rate::Real, dense_kwargs, norm_kwargs) where {F, NF}
    name = "DenseNormActDropoutBlock"
    if iszero(dropout_rate)
        if norm_layer === nothing
            return Lux.Chain(;
                dense=Lux.Dense(in_dims => out_dims, activation; dense_kwargs...), name)
        end
        return Lux.Chain(; dense=Lux.Dense(in_dims => out_dims; dense_kwargs...),
            norm=norm_layer(i, out_dims, activation; norm_kwargs...), name)
    end
    if norm_layer === nothing
        return Lux.Chain(;
            dense=Lux.Dense(in_dims => out_dims, activation; dense_kwargs...),
            dropout=Lux.Dropout(dropout_rate), name)
    end
    return Lux.Chain(; dense=Lux.Dense(in_dims => out_dims; dense_kwargs...),
        norm=norm_layer(i, out_dims, activation; norm_kwargs...),
        dropout=Lux.Dropout(dropout_rate), name)
end
