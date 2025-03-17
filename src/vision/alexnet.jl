"""
    AlexNet(; kwargs...)

Create an AlexNet model [krizhevsky2012imagenet](@citep).

## Keyword Arguments

  - `pretrained::Bool=false`: If `true`, loads pretrained weights when `LuxCore.setup` is
    called.
"""
@concrete struct AlexNet <: AbstractLuxVisionLayer
    layer
    pretrained_name::Symbol
    pretrained::Bool
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
    return AlexNet(alexnet, :alexnet, pretrained)
end
