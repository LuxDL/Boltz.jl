module Vision

using ArgCheck: @argcheck
using Compat: @compat
using ConcreteStructs: @concrete
using Random: Xoshiro

using Lux: Lux
using LuxCore: LuxCore, AbstractExplicitLayer
using NNlib: relu

using ..InitializeModels: maybe_initialize_model, INITIALIZE_KWARGS
using ..Layers: Layers, ConvBatchNormActivation, ClassTokens, ViPosEmbedding,
                VisionTransformerEncoder
using ..Utils: flatten_spatial, second_dim_mean, is_extension_loaded

include("extensions.jl")
include("vit.jl")
include("vgg.jl")

@compat(public,
    (AlexNet, ConvMixer, DenseNet, MobileNet, ResNet,
        ResNeXt, GoogLeNet, ViT, VisionTransformer, VGG))

end
