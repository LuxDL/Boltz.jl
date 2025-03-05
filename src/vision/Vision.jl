module Vision

using ArgCheck: @argcheck
using Compat: @compat
using ConcreteStructs: @concrete
using Random: Random, AbstractRNG

using Lux: Lux, Chain, Conv, BatchNorm, AdaptiveMeanPool, NoOpLayer
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer
using NNlib: relu

using ..InitializeModels: InitializeModels
using ..Layers:
    Layers,
    ConvBatchNormActivation,
    ClassTokens,
    PatchEmbedding,
    ViPosEmbedding,
    VisionTransformerEncoder
using ..Utils: second_dim_mean, is_extension_loaded

abstract type AbstractLuxVisionLayer <: AbstractLuxWrapperLayer{:layer} end

function weights_load_function(m)
    format = hasfield(typeof(m), :pretrained_format) ? m.pretrained_format : :jld2
    if format == :jld2
        return InitializeModels.load_using_jld2
    elseif format == :pth
        return InitializeModels.load_using_pickle
    else
        error("Unknown pretrained format: $(format)")
    end
end

get_weights_url(m) = hasfield(typeof(m), :weights_url) ? m.weights_url : nothing

for op in (:states, :parameters)
    fname = Symbol(:initial, op)
    fname_load = Symbol(:load, op)
    @eval function LuxCore.$(fname)(rng::AbstractRNG, model::AbstractLuxVisionLayer)
        if hasfield(typeof(model), :pretrained) && model.pretrained
            fn = weights_load_function(model)
            return InitializeModels.$(fname_load)(
                model,
                fn(
                    InitializeModels.get_pretrained_weights_path(
                        get_weights_url(model), model.pretrained_name
                    ),
                    $(string(op)),
                ),
            )
        end
        return LuxCore.$(fname)(rng, model.layer)
    end
end

include("extensions.jl")
include("alexnet.jl")
include("vit.jl")
include("vgg.jl")
include("efficientnet.jl")

@compat(
    public,
    (
        AlexNet,
        ConvMixer,
        DenseNet,
        MobileNet,
        ResNet,
        ResNeXt,
        SqueezeNet,
        GoogLeNet,
        ViT,
        VisionTransformer,
        VGG,
        WideResNet,
    )
)

end
