module Vision

using ArgCheck: @argcheck
using Compat: @compat
using ConcreteStructs: @concrete
using Random: Random, AbstractRNG
using Setfield: @set!

using Lux:
    Lux,
    Chain,
    Conv,
    Dense,
    BatchNorm,
    AdaptiveMeanPool,
    NoOpLayer,
    SamePad,
    FlattenLayer,
    SkipConnection,
    Dropout,
    WrappedFunction,
    LayerNorm,
    MaxPool
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxWrapperLayer
using NNlib: relu, swish, σ

using ..InitializeModels: InitializeModels
using ..Layers:
    Layers,
    ConvBatchNormActivation,
    ClassTokens,
    PatchEmbedding,
    ViPosEmbedding,
    VisionTransformerEncoder
using ..Utils: second_dim_mean, is_extension_loaded
using ..PytorchLoadUtils: PytorchLoadUtils

abstract type AbstractLuxVisionLayer <: AbstractLuxWrapperLayer{:layer} end

function weights_load_function(m)
    format = hasfield(typeof(m), :pretrained_format) ? m.pretrained_format : :jld2
    if format == :jld2
        return format, InitializeModels.load_using_jld2
    elseif format == :pth
        return format, InitializeModels.load_using_pickle
    else
        error("Unknown pretrained format: $(format)")
    end
end

for op in (:states, :parameters)
    fname = Symbol(:initial, op)
    fname_load = Symbol(:load_, op)
    @eval function LuxCore.$(fname)(rng::AbstractRNG, model::AbstractLuxVisionLayer)
        if hasfield(typeof(model), :pretrained) && model.pretrained
            ext, fn = weights_load_function(model)
            return InitializeModels.$(fname_load)(
                rng,
                model,
                fn(
                    InitializeModels.get_pretrained_weights_path(
                        InitializeModels.get_pretrained_weights_url(model),
                        model.pretrained_name,
                        string(ext),
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
        EfficientNet,
    )
)

end
