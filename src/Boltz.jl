module Boltz

using ArgCheck: @argcheck
using Artifacts: Artifacts, @artifact_str
using JLD2: JLD2, load
using ForwardDiff: ForwardDiff
using GPUArraysCore: GPUArraysCore
using LazyArtifacts: LazyArtifacts
using Random: Random
using Reexport: @reexport
using Statistics: mean

@reexport using Lux

@inline _is_extension_loaded(::Val) = false

# Utility Functions
include("utils.jl")
include("initialize.jl")
include("patch.jl")

# Basis Functions
include("basis.jl")

# Layers
include("layers/Layers.jl")

# Kolmogorov-Arnold Networks
include("kan/KAN.jl")

# Vision Models
include("vision/Vision.jl")

# deprecated
include("deprecated.jl")

export Basis, KAN, Layers, Vision

end
