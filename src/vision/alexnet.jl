"""
    AlexNet(; pretrained=false)

Create an AlexNet model [krizhevsky2012imagenet](@citep).

## Keyword Arguments

  - `pretrained`: Valid Options are `false`, `true`, `:DEFAULT`, `:ImageNet1K` or
    `:ImageNet1K_V1`. `:DEFAULT`, `true` and `:ImageNet1K` weights _currently_
    corresponds to `:ImageNet1K_V1`.
"""
@concrete struct AlexNet <: AbstractBoltzModel
    layer
    pretrained
end

function AlexNet(; pretrained=false)
    alexnet = Chain(;
        backbone=Chain(
            Conv((11, 11), 3 => 64, relu; stride=4, pad=2),
            MaxPool((3, 3); stride=2),
            Conv((5, 5), 64 => 192, relu; pad=2),
            MaxPool((3, 3); stride=2),
            Conv((3, 3), 192 => 384, relu; pad=1),
            Conv((3, 3), 384 => 256, relu; pad=1),
            Conv((3, 3), 256 => 256, relu; pad=1),
            MaxPool((3, 3); stride=2),
        ),
        classifier=Chain(
            AdaptiveMeanPool((6, 6)),
            FlattenLayer(),
            Dropout(0.5f0),
            Dense(256 * 6 * 6 => 4096, relu),
            Dropout(0.5f0),
            Dense(4096 => 4096, relu),
            Dense(4096 => 1000),
        ),
    )
    return AlexNet(alexnet, get_alexnet_pretrained_weights(pretrained))
end

abstract type AbstractAlexNetWeights <: PretrainedWeights.ArtifactsPretrainedWeight end

function get_alexnet_pretrained_weights(pretrained::Bool)
    !pretrained && return nothing
    return get_alexnet_pretrained_weights(:DEFAULT)
end

function get_alexnet_pretrained_weights(name::Union{String,Symbol})
    name = Symbol(name)
    @argcheck name in (:ImageNet1K_V1, :ImageNet1K, :DEFAULT)
    name == :DEFAULT && (name = :ImageNet1K_V1)
    name == :ImageNet1K && (name = :ImageNet1K_V1)
    name == :ImageNet1K_V1 && return AlexNet_Weights_ImageNet1K_V1()
    error("Unknown pretrained weights name: $(name))")
end

get_alexnet_pretrained_weights(W::AbstractAlexNetWeights) = W

struct AlexNet_Weights_ImageNet1K_V1 <: AbstractAlexNetWeights end

PretrainedWeights.load_with(::Val{:JLD2}, ::AlexNet_Weights_ImageNet1K_V1) = true

PretrainedWeights.get_artifact_name(::AlexNet_Weights_ImageNet1K_V1) = "alexnet"
