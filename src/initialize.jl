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

function get_pretrained_weights_path(url, name::Symbol)
    return get_pretrained_weights_path(url, string(name))
end

function get_pretrained_weights_path(::Nothing, name::String)
    try
        dir = @artifact_str(name)
        return joinpath(dir, "$(name).jld2")
    catch err
        err isa ErrorException &&
            throw(ArgumentError("no pretrained weights available for `$name`"))
        rethrow(err)
    end
end

function get_pretrained_weights_path(url::String, name::String)
    scratch_dir = @get_scratch!(name)
    filename = basename(url)
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

loadparameters(model, x) = loadparameters_fallback(x)

loadparameters_fallback(x) = x

loadstates(model, x) = loadstates_fallback(x)

function loadstates_fallback(x)
    return fmap(x) do xᵢ
        xᵢ isa SerializedRNG && return Random.default_rng()
        return xᵢ
    end
end

end
