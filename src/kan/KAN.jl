module KAN

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArgCheck: @argcheck
    using ..Boltz: Boltz, Basis
    using ConcreteStructs: @concrete
    using Lux: Lux
    using LuxCore: LuxCore, AbstractExplicitLayer
    using LuxLib: fused_dense_bias_activation
    using NNlib: NNlib
    using Random: Random, AbstractRNG
    using WeightInitializers: randn32, kaiming_normal, zeros32
end

include("dense.jl")

end
