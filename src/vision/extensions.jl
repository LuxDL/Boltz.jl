"""
    ResNet(depth::Int; pretrained::Bool=false)

Create a ResNet model [he2016deep](@citep).

## Arguments

  - `depth::Int`: The depth of the ResNet model. Must be one of 18, 34, 50, 101, or 152.

## Keyword Arguments

  - `pretrained::Bool=false`: If `true`, loads pretrained weights when `LuxCore.setup` is
    called.
"""
function ResNet end

"""
    ResNeXt(depth::Int; cardinality=32, base_width=nothing, pretrained::Bool=false)

Create a ResNeXt model [xie2017aggregated](@citep).

## Arguments

  - `depth::Int`: The depth of the ResNeXt model. Must be one of 50, 101, or 152.

## Keyword Arguments

  - `pretrained::Bool=false`: If `true`, loads pretrained weights when `LuxCore.setup` is
    called.
  - `cardinality`: The cardinality of the ResNeXt model. Defaults to 32.
  - `base_width`: The base width of the ResNeXt model. Defaults to 8 for depth 101 and 4
    otherwise.
"""
function ResNeXt end

"""
    GoogLeNet(; pretrained::Bool=false)

Create a GoogLeNet model [szegedy2015going](@citep).

## Keyword Arguments

  - `pretrained::Bool=false`: If `true`, loads pretrained weights when `LuxCore.setup` is
    called.
"""
function GoogLeNet end

"""
    DenseNet(depth::Int; pretrained::Bool=false)

Create a DenseNet model [huang2017densely](@citep).

## Arguments

  - `depth::Int`: The depth of the DenseNet model. Must be one of 121, 161, 169, or 201.

## Keyword Arguments

  - `pretrained::Bool=false`: If `true`, loads pretrained weights when `LuxCore.setup` is
    called.
"""
function DenseNet end

"""
    MobileNet(name::Symbol; pretrained::Bool=false)

Create a MobileNet model
[howard2017mobilenets, sandler2018mobilenetv2, howard2019searching](@citep).

## Arguments

  - `name::Symbol`: The name of the MobileNet model. Must be one of `:v1`, `:v2`,
    `:v3_small`, or `:v3_large`.

## Keyword Arguments

  - `pretrained::Bool=false`: If `true`, loads pretrained weights when `LuxCore.setup` is
    called.
"""
function MobileNet end

"""
    ConvMixer(name::Symbol; pretrained::Bool=false)

Create a ConvMixer model [trockman2022patches](@citep).

## Arguments

  - `name::Symbol`: The name of the ConvMixer model. Must be one of `:base`, `:small`, or
    `:large`.

## Keyword Arguments

  - `pretrained::Bool=false`: If `true`, loads pretrained weights when `LuxCore.setup` is
    called.
"""
function ConvMixer end

"""
    SqueezeNet(; pretrained::Bool=false)

Create a SqueezeNet model [iandola2016squeezenetalexnetlevelaccuracy50x](@citep).

## Keyword Arguments

  - `pretrained::Bool=false`: If `true`, loads pretrained weights when `LuxCore.setup` is
    called.
"""
function SqueezeNet end

"""
    WideResNet(depth::Int; pretrained::Bool=false)

Create a WideResNet model [zagoruyko2017wideresidualnetworks](@citep).

## Arguments

  - `depth::Int`: The depth of the WideResNet model. Must be one of 18, 34, 50, 101, or 152.

## Keyword Arguments

  - `pretrained::Bool=false`: If `true`, loads pretrained weights when `LuxCore.setup` is
    called.
"""
function WideResNet end

@concrete struct MetalheadWrapperLayer <: AbstractBoltzModel
    layer
end

for f in [
    :ResNet,
    :ResNeXt,
    :GoogLeNet,
    :DenseNet,
    :MobileNet,
    :ConvMixer,
    :SqueezeNet,
    :WideResNet,
]
    f_metalhead = Symbol(f, :Metalhead)
    @eval begin
        function $(f_metalhead) end
        function $(f)(args...; pretrained=false, kwargs...)
            if !is_extension_loaded(Val(:Metalhead))
                error("`Metalhead.jl` is not loaded. Please load `Metalhead.jl` to use \
                       this function.")
            end
            return MetalheadWrapperLayer(
                $(f_metalhead)(
                    args...;
                    pretrained=convert_metalhead_pretrained_to_bool(pretrained),
                    kwargs...,
                ),
            )
        end
    end
end

function convert_metalhead_pretrained_to_bool(name::Union{String,Symbol})
    name = Symbol(name)
    @argcheck name in (:ImageNet1K, :DEFAULT)
    return true
end
convert_metalhead_pretrained_to_bool(name::Bool) = name
