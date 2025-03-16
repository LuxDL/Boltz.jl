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
    alexnet = Lux.Chain(;
        backbone=Lux.Chain(
            Lux.Conv((11, 11), 3 => 64, relu; stride=4, pad=2),
            Lux.MaxPool((3, 3); stride=2),
            Lux.Conv((5, 5), 64 => 192, relu; pad=2),
            Lux.MaxPool((3, 3); stride=2),
            Lux.Conv((3, 3), 192 => 384, relu; pad=1),
            Lux.Conv((3, 3), 384 => 256, relu; pad=1),
            Lux.Conv((3, 3), 256 => 256, relu; pad=1),
            Lux.MaxPool((3, 3); stride=2),
        ),
        classifier=Lux.Chain(
            Lux.AdaptiveMeanPool((6, 6)),
            Lux.FlattenLayer(),
            Lux.Dropout(0.5f0),
            Lux.Dense(256 * 6 * 6 => 4096, relu),
            Lux.Dropout(0.5f0),
            Lux.Dense(4096 => 4096, relu),
            Lux.Dense(4096 => 1000),
        ),
    )
    return AlexNet(alexnet, :alexnet, pretrained)
end
