module InitializeModels

using ArgCheck: @argcheck
using Artifacts: Artifacts, @artifact_str
using LazyArtifacts: LazyArtifacts
using Random: Random

using LuxCore: LuxCore

using ..Utils: is_extension_loaded, unwrap_val

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

function load_using_jld2(args...; kwargs...)
    if !is_extension_loaded(Val(:JLD2))
        error("`JLD2.jl` is not loaded. Please load it before trying to load pretrained \
               weights.")
    end
    return load_using_jld2_internal(args...; kwargs...)
end

function load_using_jld2_internal end

end
