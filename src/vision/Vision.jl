module Vision

using ArgCheck: @argcheck
using Compat: @compat
using ConcreteStructs: @concrete
using Random: Random, AbstractRNG

using Lux: Lux
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxWrapperLayer
using NNlib: relu

using ..InitializeModels: InitializeModels
using ..Layers: Layers, ConvBatchNormActivation, ClassTokens, PatchEmbedding,
                ViPosEmbedding, VisionTransformerEncoder
using ..Utils: second_dim_mean, is_extension_loaded

abstract type AbstractLuxVisionLayer <: AbstractLuxWrapperLayer{:layer} end

for op in (:states, :parameters)
    fname = Symbol(:initial, op)
    fname_load = Symbol(:load, op)
    @eval function LuxCore.$(fname)(rng::AbstractRNG, model::AbstractLuxVisionLayer)
        if hasfield(typeof(model), :pretrained) && model.pretrained
            path = InitializeModels.get_pretrained_weights_path(model.pretrained_name)
            jld2_loaded_obj = InitializeModels.load_using_jld2(
                joinpath(path, "$(model.pretrained_name).jld2"), $(string(op)))
            return InitializeModels.$(fname_load)(jld2_loaded_obj)
        end
        return LuxCore.$(fname)(rng, model.layer)
    end
end

include("extensions.jl")
include("alexnet.jl")
include("vit.jl")
include("vgg.jl")

@compat(public,
    (AlexNet, ConvMixer, DenseNet, MobileNet, ResNet,
        ResNeXt, GoogLeNet, ViT, VisionTransformer, VGG))

end
