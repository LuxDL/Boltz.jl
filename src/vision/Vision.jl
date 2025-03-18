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

using ..PretrainedWeights: PretrainedWeights
using ..InitializeModels: InitializeModels, AbstractBoltzModel
using ..Layers:
    Layers,
    ConvBatchNormActivation,
    ClassTokens,
    PatchEmbedding,
    ViPosEmbedding,
    VisionTransformerEncoder
using ..Utils: second_dim_mean, is_extension_loaded
using ..PytorchLoadUtils: PytorchLoadUtils

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
