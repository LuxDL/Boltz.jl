module Layers

using ArgCheck: @argcheck
using ADTypes: AutoForwardDiff, AutoZygote
using Compat: @compat
using ConcreteStructs: @concrete
using ChainRulesCore: ChainRulesCore, @non_differentiable, @ignore_derivatives
using Markdown: @doc_str
using Random: AbstractRNG
using Static: Static

using ForwardDiff: ForwardDiff

using Lux: Lux, LuxOps, StatefulLuxLayer
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer
using MLDataDevices: get_device, CPUDevice
using NNlib: NNlib
using WeightInitializers: zeros32, randn32

using ..Utils: DataTransferBarrier, fast_chunk, should_type_assert, mapreduce_stack,
               unwrap_val, safe_kron, is_extension_loaded, flatten_spatial

const CRC = ChainRulesCore

const NORM_LAYER_DOC = "Function with signature `f(i::Integer, dims::Integer, act::F; \
                        kwargs...)`. `i` is the location of the layer in the model, \
                        `dims` is the channel dimension of the input, and `act` is the \
                        activation function. `kwargs` are forwarded from the `norm_kwargs` \
                        input, The function should return a normalization layer. Defaults \
                        to `nothing`, which means no normalization layer is used"

include("attention.jl")
include("conv_norm_act.jl")
include("dynamic_expressions.jl")
include("encoder.jl")
include("embeddings.jl")
include("hamiltonian.jl")
include("lyapunov_net.jl")
include("mlp.jl")
include("spline.jl")
include("tensor_product.jl")

@compat(public,
    (ClassTokens, ConvBatchNormActivation, ConvNormActivation, DynamicExpressionsLayer,
        HamiltonianNN, MultiHeadSelfAttention, MLP, PatchEmbedding, PeriodicEmbedding,
        PositiveDefinite, ShiftTo, SplineLayer, TensorProductLayer, ViPosEmbedding,
        VisionTransformerEncoder))

end
