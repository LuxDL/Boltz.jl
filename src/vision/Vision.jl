module Vision

using ArgCheck: @argcheck
using Lux: Lux
using LuxCore: LuxCore, AbstractExplicitLayer
using NNlib: relu
using Random: Xoshiro

using ..Boltz: _is_extension_loaded, __maybe_initialize_model, Layers,
               INITIALIZE_KWARGS
using ..Utils: flatten_spatial, second_dim_mean

include("extensions.jl")
include("vit.jl")
include("vgg.jl")

end
