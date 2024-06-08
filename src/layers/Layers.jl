module Layers

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArgCheck: @argcheck
    using ..Boltz: _fast_chunk
    using ConcreteStructs: @concrete
    using ChainRulesCore: ChainRulesCore
    using Lux: Lux
    using LuxCore: LuxCore, AbstractExplicitLayer
    using NNlib: NNlib
    using Random: AbstractRNG
    using WeightInitializers: zeros32, randn32
end

const CRC = ChainRulesCore

include("conv_norm_act.jl")
include("attention.jl")
include("encoder.jl")
include("embeddings.jl")

end
