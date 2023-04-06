module Boltz

using CUDA, Lux, NNlib, Random, Statistics
# Loading Pretained Weights
using Artifacts, JLD2, LazyArtifacts
# AD Support
import ChainRulesCore as CRC

# Extensions
if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    @static if !isdefined(Base, :get_extension)
        # Import Flux Models
        @require Metalhead="dbeba491-748d-5e0e-a39e-b530a07fa0cc" begin include("../ext/BoltzFluxMetalheadExt.jl") end
    end
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

# Exports
export alexnet, convmixer, densenet, googlenet, mobilenet, resnet, resnext, vgg,
       vision_transformer

end
