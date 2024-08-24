"""
    AlexNet(; kwargs...)

Create an AlexNet model [krizhevsky2012imagenet](@citep).

## Keyword Arguments

$(INITIALIZE_KWARGS)
"""
function AlexNet end

"""
    ResNet(depth::Int; kwargs...)

Create a ResNet model [he2016deep](@citep).

## Arguments

  * `depth::Int`: The depth of the ResNet model. Must be one of 18, 34, 50, 101, or 152.

## Keyword Arguments

$(INITIALIZE_KWARGS)
"""
function ResNet end

"""
    ResNeXt(depth::Int; kwargs...)

Create a ResNeXt model [xie2017aggregated](@citep).

## Arguments

  * `depth::Int`: The depth of the ResNeXt model. Must be one of 50, 101, or 152.

## Keyword Arguments

$(INITIALIZE_KWARGS)
"""
function ResNeXt end

"""
    GoogLeNet(; kwargs...)

Create a GoogLeNet model [szegedy2015going](@citep).

## Keyword Arguments

$(INITIALIZE_KWARGS)
"""
function GoogLeNet end

"""
    DenseNet(depth::Int; kwargs...)

Create a DenseNet model [huang2017densely](@citep).

## Arguments

  * `depth::Int`: The depth of the DenseNet model. Must be one of 121, 161, 169, or 201.

## Keyword Arguments

$(INITIALIZE_KWARGS)
"""
function DenseNet end

"""
    MobileNet(name::Symbol; kwargs...)

Create a MobileNet model
[howard2017mobilenets, sandler2018mobilenetv2, howard2019searching](@citep).

## Arguments

  * `name::Symbol`: The name of the MobileNet model. Must be one of `:v1`, `:v2`,
    `:v3_small`, or `:v3_large`.

## Keyword Arguments

$(INITIALIZE_KWARGS)
"""
function MobileNet end

"""
    ConvMixer(name::Symbol; kwargs...)

Create a ConvMixer model [trockman2022patches](@citep).

## Arguments

  * `name::Symbol`: The name of the ConvMixer model. Must be one of `:base`, `:small`, or
    `:large`.

## Keyword Arguments

$(INITIALIZE_KWARGS)
"""
function ConvMixer end

for f in [:AlexNet, :ResNet, :ResNeXt, :GoogLeNet, :DenseNet, :MobileNet, :ConvMixer]
    f_metalhead = Symbol(f, :Metalhead)
    @eval begin
        function $(f_metalhead) end
        function $(f)(args...; kwargs...)
            if !is_extension_loaded(Val(:Metalhead))
                error("`Metalhead.jl` is not loaded. Please load `Metalhead.jl` to use \
                       this function.")
            end
            $(f_metalhead)(args...; kwargs...)
        end
    end
end
