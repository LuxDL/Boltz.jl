module Boltz

using PrecompileTools: @recompile_invalidations
using Reexport: @reexport

@recompile_invalidations begin
    using ArgCheck: @argcheck
    using Artifacts: Artifacts, @artifact_str
    using JLD2: JLD2, load
    using GPUArraysCore: GPUArraysCore
    using LazyArtifacts: LazyArtifacts
    using Lux
    using Random: Random
    using Statistics: mean
end

@reexport using Lux

@inline _is_extension_loaded(::Val) = false

# Utility Functions
include("utils.jl")
include("initialize.jl")
include("patch.jl")

# Layers
include("layers/Layers.jl")

# Vision Models
include("vision/Vision.jl")

# deprecated
include("deprecated.jl")

export Layers, Vision

end
