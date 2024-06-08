module Vision

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArgCheck: @argcheck
    using ..Boltz: _is_extension_loaded, _flatten_spatial, _seconddimmean,
                   __maybe_initialize_model, Layers, INITIALIZE_KWARGS
    using Lux: Lux
    using LuxCore: LuxCore
    using NNlib: relu
    using Random: Xoshiro
end

include("extensions.jl")
include("vit.jl")
include("vgg.jl")

end
