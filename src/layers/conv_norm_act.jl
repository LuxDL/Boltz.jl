"""
    ConvBatchNormActivation(kernel_size::Dims, (in_filters, out_filters)::Pair{Int, Int},
        depth::Int, act::F; use_norm::Bool=true, conv_kwargs=(;),
        norm_kwargs=(;), flatten_model=false) where {F}

Construct a Chain of convolutional layers with batch normalization and activation functions.

3# Arguments

  - `kernel_size`: size of the convolutional kernel
  - `in_filters`: number of input filters
  - `out_filters`: number of output filters
  - `depth`: number of convolutional layers
  - `act`: activation function

## Keyword Arguments

  - `use_norm`: set to `true` to include batch normalization after each convolution
  - `conv_kwargs`: keyword arguments for the convolutional layers
  - `norm_kwargs`: keyword arguments for the batch normalization layers
  - `flatten_model`: set to `true` construct a flat chain without internal chains (not
    recommended)
"""
function ConvBatchNormActivation(
        kernel_size::Dims, (in_filters, out_filters)::Pair{Int, Int},
        depth::Int, act::F; use_norm::Bool=true, conv_kwargs=(;),
        norm_kwargs=(;), flatten_model=false) where {F}
    layers = []
    for _ in 1:depth
        if use_norm
            push!(layers,
                Lux.Chain(Lux.Conv(kernel_size, in_filters => out_filters; conv_kwargs...),
                    Lux.BatchNorm(out_filters, act; norm_kwargs...); name="ConvBN"))
        else
            push!(layers,
                Lux.Conv(kernel_size, in_filters => out_filters, act; conv_kwargs...))
        end
        in_filters = out_filters
    end
    return Lux.Chain(
        layers...; disable_optimizations=!flatten_model, name="ConvBatchNormActivation")
end
