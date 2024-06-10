module KAN

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArgCheck: @argcheck
    using ..Boltz: Boltz, Basis
    using ConcreteStructs: @concrete
    using LuxCore: AbstractExplicitLayer, LuxCore
    using NNlib: NNlib
    using Random: Random, AbstractRNG
    using WeightInitializers: randn32, kaiming_normal
end

include("dense.jl")

end
