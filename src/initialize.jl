module InitializeModels

using ArgCheck: @argcheck
using Artifacts: Artifacts, @artifact_str
using Downloads: Downloads
using Functors: fmap
using LazyArtifacts: LazyArtifacts
using Random: Random
using Scratch: @get_scratch!

using LuxCore: LuxCore

using ..Utils: is_extension_loaded

get_pretrained_weights_url(_) = nothing

function get_pretrained_weights_path(url, name::Symbol, ext::String)
    return get_pretrained_weights_path(url, string(name), ext)
end

function get_pretrained_weights_path(::Nothing, name::String, ext::String)
    try
        return joinpath(@artifact_str(name), "$(name).$(ext)")
    catch err
        err isa ErrorException &&
            throw(ArgumentError("no pretrained weights available for `$name`"))
        rethrow(err)
    end
end

function get_pretrained_weights_path(url::String, name::String, ext::String)
    scratch_dir = @get_scratch!(name)
    filename = basename(url)
    @assert endswith(filename, ext) "Mismatched pretrained weights extension. Got \
                                     `$filename`, expected `$ext`"
    weights_path = joinpath(scratch_dir, filename)
    !isfile(weights_path) && Downloads.download(url, weights_path)
    return weights_path
end

function load_using_jld2(args...; kwargs...)
    if !is_extension_loaded(Val(:JLD2))
        error("`JLD2.jl` is not loaded. Please load it before trying to load pretrained \
               weights.")
    end
    return load_using_jld2_internal(args...; kwargs...)
end

function load_using_jld2_internal end

function load_using_pickle(args...; kwargs...)
    if !is_extension_loaded(Val(:Pickle))
        error("`Pickle.jl` is not loaded. Please load it before trying to load pretrained \
               weights.")
    end
    return load_using_pickle_internal(args...; kwargs...)
end

function load_using_pickle_internal end

struct SerializedRNG end

function remove_rng_from_structure(x)
    return fmap(x) do xᵢ
        xᵢ isa Random.AbstractRNG && return SerializedRNG()
        return xᵢ
    end
end

load_parameters(rng, model, ps) = loadparameters_fallback(ps)

load_parameters_fallback(ps) = ps

load_states(rng, model, st) = load_states_fallback(st)

function load_states_fallback(st)
    return fmap(st) do stᵢ
        stᵢ isa SerializedRNG && return Random.default_rng()
        return stᵢ
    end
end

end
