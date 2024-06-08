__get_pretrained_weights_path(name::Symbol) = __get_pretrained_weights_path(string(name))
function __get_pretrained_weights_path(name::String)
    try
        return @artifact_str(name)
    catch err
        err isa ErrorException &&
            throw(ArgumentError("no pretrained weights available for `$name`"))
        rethrow(err)
    end
end

const INITIALIZE_KWARGS = """
  * `pretrained::Bool=false`: If `true`, returns a pretrained model.
  * `rng::Union{Nothing, AbstractRNG}=nothing`: Random number generator.
  * `seed::Int=0`: Random seed.
  * `initialized::Val{Bool}=Val(true)`: If `Val(true)`, returns
    `(model, parameters, states)`, otherwise just `model`.
"""

function __initialize_model(
        name::Symbol, model; pretrained::Bool=false, rng=nothing, seed=0, kwargs...)
    if pretrained
        path = __get_pretrained_weights_path(name)
        ps = load(joinpath(path, "$name.jld2"), "parameters")
        st = load(joinpath(path, "$name.jld2"), "states")
        return ps, st
    end
    if rng === nothing
        rng = Random.default_rng()
        Random.seed!(rng, seed)
    end
    return LuxCore.setup(rng, model)
end

function __maybe_initialize_model(name::Symbol, model; pretrained=false,
        initialized::Union{Val, Bool}=Val(true), kwargs...)
    @argcheck !pretrained || __unwrap_val(initialized)
    __unwrap_val(initialized) || return model
    ps, st = __initialize_model(name, model; pretrained, kwargs...)
    return model, ps, st
end

@inline __unwrap_val(::Val{T}) where {T} = T
@inline __unwrap_val(T) = T
