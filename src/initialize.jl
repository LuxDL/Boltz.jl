module InitializeModels

using ArgCheck: @argcheck
using Artifacts: Artifacts, @artifact_str
using JLD2: JLD2
using Random: Random

using LuxCore: LuxCore

using ..Utils: unwrap_val

get_pretrained_weights_path(name::Symbol) = get_pretrained_weights_path(string(name))
function get_pretrained_weights_path(name::String)
    try
        return @artifact_str(name)
    catch err
        err isa ErrorException &&
            throw(ArgumentError("no pretrained weights available for `$name`"))
        rethrow(err)
    end
end

function initialize_model(
        name::Symbol, model; pretrained::Bool=false, rng=nothing, seed=0, kwargs...)
    if pretrained
        path = get_pretrained_weights_path(name)
        ps = JLD2.load(joinpath(path, "$name.jld2"), "parameters")
        st = JLD2.load(joinpath(path, "$name.jld2"), "states")
        return ps, st
    end
    if rng === nothing
        rng = Random.default_rng()
        Random.seed!(rng, seed)
    end
    return LuxCore.setup(rng, model)
end

function maybe_initialize_model(name::Symbol, model; pretrained=false,
        initialized::Union{Val, Bool}=Val(true), kwargs...)
    @argcheck !pretrained || unwrap_val(initialized)
    unwrap_val(initialized) || return model
    ps, st = initialize_model(name, model; pretrained, kwargs...)
    return model, ps, st
end

const INITIALIZE_KWARGS = """
  * `pretrained::Bool=false`: If `true`, returns a pretrained model.
  * `rng::Union{Nothing, AbstractRNG}=nothing`: Random number generator.
  * `seed::Int=0`: Random seed.
  * `initialized::Val{Bool}=Val(true)`: If `Val(true)`, returns
    `(model, parameters, states)`, otherwise just `model`.
"""

end
