module Layers

using ArgCheck: @argcheck
using ADTypes: AutoForwardDiff, AutoZygote
using Compat: @compat
using ConcreteStructs: @concrete
using ChainRulesCore: @non_differentiable
using Markdown: @doc_str
using Random: AbstractRNG

using ForwardDiff: ForwardDiff

using Lux: Lux, LuxOps, StatefulLuxLayer
using LuxCore: LuxCore, AbstractExplicitLayer, AbstractExplicitContainerLayer
using NNlib: NNlib
using WeightInitializers: zeros32, randn32

using ..Utils: fast_chunk, should_type_assert, mapreduce_stack, unwrap_val, safe_kron,
               is_extension_loaded

const NORM_LAYER_DOC = "Function with signature `f(i::Integer, dims::Integer, act::F; \
                        kwargs...)`. `i` is the location of the layer in the model, \
                        `dims` is the channel dimension of the input, and `act` is the \
                        activation function. `kwargs` are forwarded from the `norm_kwargs` \
                        input, The function should return a normalization layer. Defaults \
                        to `nothing`, which means no normalization layer is used"

include("attention.jl")
include("conv_norm_act.jl")
include("encoder.jl")
include("embeddings.jl")
include("hamiltonian.jl")
include("mlp.jl")
include("spline.jl")
include("tensor_product.jl")

@compat(public,
    (ClassTokens, ConvBatchNormActivation, ConvNormActivation, HamiltonianNN,
        MultiHeadSelfAttention, MLP, SplineLayer, TensorProductLayer, ViPosEmbedding,
        VisionTransformerEncoder))

end
