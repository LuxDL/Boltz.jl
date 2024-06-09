"""
    ConvNormActivation(kernel_size::Dims, in_chs::Integer, hidden_chs::Dims{N},
        activation; norm_layer=nothing, conv_kwargs=(;), norm_kwargs=(;),
        last_layer_activation::Bool=false, flatten_model::Bool=false) where {N}

Construct a Chain of convolutional layers with normalization and activation functions.

## Arguments

  - `kernel_size`: size of the convolutional kernel
  - `in_chs`: number of input channels
  - `hidden_chs`: dimensions of the hidden layers
  - `activation`: activation function

## Keyword Arguments

  - `norm_layer`: $(NORM_LAYER_DOC)
  - `conv_kwargs`: keyword arguments for the convolutional layers
  - `norm_kwargs`: keyword arguments for the normalization layers
  - `last_layer_activation`: set to `true` to apply the activation function to the last
    layer

## Internal Keyword Arguments

Don't rely on these, they are for internal use only.

  - `flatten_model`: set to `true` construct a flat chain without internal chains (not
    recommended)
"""
function ConvNormActivation(
        kernel_size::Dims, in_chs::Integer, hidden_chs::Dims{N}, activation::F=NNlib.relu;
        norm_layer::NF=nothing, conv_kwargs=(;), norm_kwargs=(;),
        flatten_model::Bool=false, last_layer_activation::Bool=false) where {N, F, NF}
    layers = Vector{AbstractExplicitLayer}(undef, N)
    for (i, out_chs) in enumerate(hidden_chs)
        act = i != N ? activation : (last_layer_activation ? activation : identity)
        layers[i] = __conv_norm_act(
            i, kernel_size, in_chs => out_chs, act, norm_layer, conv_kwargs, norm_kwargs)
        in_chs = out_chs
    end
    flatten_model && return Lux.Chain(layers...; name="ConvNormActivation")
    inner_blocks = NamedTuple{ntuple(i -> Symbol(:block, i), N)}(layers)
    return Lux.Chain(inner_blocks; name="ConvNormActivation", disable_optimizations=true)
end

@inline function __conv_norm_act(
        i::Integer, kernel_size::Dims, (in_chs, out_chs)::Pair{<:Integer, <:Integer},
        activation::F, norm_layer::NF, conv_kwargs, norm_kwargs) where {F, NF}
    name = "ConvNormActBlock"
    norm_layer === nothing && return Lux.Chain(;
        conv=Lux.Conv(kernel_size, in_chs => out_chs, activation; conv_kwargs...),
        name)
    return Lux.Chain(; conv=Lux.Conv(kernel_size, in_chs => out_chs; conv_kwargs...),
        norm=norm_layer(i, out_chs, activation; norm_kwargs...), name)
end

"""
    ConvBatchNormActivation(kernel_size::Dims, (in_filters, out_filters)::Pair{Int, Int},
        depth::Int, act::F; use_norm::Bool=true, conv_kwargs=(;),
        last_layer_activation::Bool=true, norm_kwargs=(;), flatten_model=false) where {F}

This function is a convenience wrapper around [`ConvNormActivation`](@ref) that constructs a
chain with `norm_layer` set to `Lux.BatchNorm` if `use_norm` is `true` and `nothing`
otherwise. In most cases, users should use [`ConvNormActivation`](@ref) directly for a more
flexible interface.
"""
function ConvBatchNormActivation(
        kernel_size::Dims, (in_filters, out_filters)::Pair{Int, Int},
        depth::Int, act::F; use_norm::Bool=true, kwargs...) where {F}
    hidden_chs = ntuple(Returns(out_filters), depth)
    return ConvNormActivation(kernel_size,
        in_filters,
        hidden_chs,
        act;
        norm_layer=use_norm ?
                   (i, chs, bn_act; kwargs...) -> Lux.BatchNorm(chs, bn_act; kwargs...) :
                   nothing,
        last_layer_activation=true,
        kwargs...)
end
