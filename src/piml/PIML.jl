module PIML

using ArgCheck: @argcheck
using Compat: @compat
using ConcreteStructs: @concrete
using Random: Random, AbstractRNG
using Functors: fmap
using Setfield: @set!

using Lux: Lux
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer
using NNlib: NNlib
using WeightInitializers: truncated_normal, zeros32, randn32

using ..Layers: PhysicsSelfAttentionIrregularMesh

include("transolver.jl")

@compat(public, (Transolver,))

end
