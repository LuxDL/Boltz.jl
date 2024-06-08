module Layers

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArgCheck: @argcheck
    using ADTypes: AutoForwardDiff, AutoZygote
    using ..Boltz: _fast_chunk, _should_type_assert
    using ConcreteStructs: @concrete
    using ChainRulesCore: ChainRulesCore
    using Lux: Lux, StatefulLuxLayer
    using LuxCore: LuxCore, AbstractExplicitLayer, AbstractExplicitContainerLayer
    using NNlib: NNlib
    using Random: AbstractRNG
    using WeightInitializers: zeros32, randn32
end

const CRC = ChainRulesCore

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

end
