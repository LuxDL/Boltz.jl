module Boltz

using CUDA, Lux, NNlib, Random, Statistics
# Loading Pretained Weights
using Artifacts, JLD2, LazyArtifacts
# AD Support
import ChainRulesCore as CRC

# Extensions
using PackageExtensionCompat
function __init__()
    @require_extensions
end

# Define functions. Methods defined in files or in extensions later
for f in (:alexnet,
    :convmixer,
    :densenet,
    :googlenet,
    :mobilenet,
    :resnet,
    :resnext,
    :vgg,
    :vision_transformer)
    @eval function $(f) end
end

# Utility Functions
include("utils.jl")

# Vision Models
include("vision/vit.jl")
include("vision/vgg.jl")

# Exports
export alexnet,
    convmixer, densenet, googlenet, mobilenet, resnet, resnext, vgg, vision_transformer

end
