module Vision

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArgCheck: @argcheck
    using ..Boltz: _flatten_spatial, _seconddimmean, __maybe_initialize_model, Layers,
                   INITIALIZE_KWARGS
    using Lux: Lux
    using LuxCore: LuxCore
    using NNlib: relu
    using Random: Xoshiro
end

# Define functions. Methods defined in extensions later
for f in (:AlexNet, :ConvMixer, :DenseNet, :GoogLeNet, :MobileNet, :ResNet, :ResNeXt)
    @eval function $(f) end
end

include("vit.jl")
include("vgg.jl")

end
