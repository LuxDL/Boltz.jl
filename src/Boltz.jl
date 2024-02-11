module Boltz

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using Lux, Random, Statistics, Artifacts, JLD2, LazyArtifacts
    import ChainRulesCore as CRC
    import GPUArraysCore
end

# Define functions. Methods defined in files or in extensions later
for f in (:alexnet, :convmixer, :densenet, :googlenet, :mobilenet, :resnet, :resnext, :vgg,
    :vision_transformer)
    @eval function $(f) end
end

# Utility Functions
include("utils.jl")

# Vision Models
include("vision/vit.jl")
include("vision/vgg.jl")

# Hacky Patch for loading pretrained models
@static if VERSION ≥ v"1.10-"
    function Base.convert(::Type{Random.Xoshiro},
            x::JLD2.ReconstructedStatic{Symbol("Random.Xoshiro"), (:s0, :s1, :s2, :s3),
                NTuple{4, UInt64}})
        return Random.Xoshiro(x.s0, x.s1, x.s2, x.s3)
    end
    function Base.convert(::Type{Random.Xoshiro},
            x::JLD2.ReconstructedStatic{:Xoshiro, (:s0, :s1, :s2, :s3), NTuple{4, UInt64}})
        return Random.Xoshiro(x.s0, x.s1, x.s2, x.s3)
    end
end

# Exports
export alexnet,
       convmixer, densenet, googlenet, mobilenet, resnet, resnext, vgg, vision_transformer

end
