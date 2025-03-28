module InitializeModels

using ArgCheck: @argcheck
using Functors: fmap
using Random: Random, AbstractRNG

using LuxCore: LuxCore, AbstractLuxWrapperLayer

using ..PretrainedWeights: PretrainedWeights
using ..Utils: is_extension_loaded

abstract type AbstractBoltzModel <: AbstractLuxWrapperLayer{:layer} end

for op in (:states, :parameters)
    fname = Symbol(:initial, op)
    fname_load = Symbol(:load_, op)
    @eval function LuxCore.$(fname)(rng::AbstractRNG, model::AbstractBoltzModel)
        if hasfield(typeof(model), :pretrained) && model.pretrained !== nothing
            ext = PretrainedWeights.checkpoint_extension(model.pretrained)
            path = PretrainedWeights.download_and_get_checkpoint_path(model.pretrained)
            loaded_weights = if PretrainedWeights.load_with(Val(:JLD2), model.pretrained)
                load_using_jld2(path, $(string(op)))
            elseif PretrainedWeights.load_with(Val(:Pickle), model.pretrained)
                load_using_pickle(path)
            elseif PretrainedWeights.load_with(Val(:SafeTensors), model.pretrained)
                load_using_safetensors(path)
            else
                error("Unknown pretrained weights format: $(ext)")
            end
            return $(fname_load)(rng, model, loaded_weights)
        end
        return LuxCore.$(fname)(rng, model.layer)
    end
end

# Formats and Packages for loading pretrained weights. These are defined in extensions
# to avoid heavy dependencies.
for (pkg, lpkg) in ((:JLD2, :jld2), (:Pickle, :pickle), (:SafeTensors, :safetensors))
    fname = Symbol(:load_using_, lpkg)
    fname_internal = Symbol(:load_using_, lpkg, :_internal)
    @eval begin
        function $(fname)(args...; kwargs...)
            if !is_extension_loaded(Val($(Meta.quot(pkg))))
                error("`$($pkg).jl` is not loaded. Please load it before trying to load \
                       pretrained weights.")
            end
            return $(fname_internal)(args...; kwargs...)
        end

        function $(fname_internal) end
    end
end

# Load & Save Parameters & States. Models has overload `load_parameters` and `load_states`
# to provide custom loaders (See EfficientNet for example).
struct SerializedRNG end

function remove_rng_from_structure(x)
    return fmap(x) do xᵢ
        xᵢ isa Random.AbstractRNG && return SerializedRNG()
        return xᵢ
    end
end

load_parameters(rng, model, ps) = load_parameters_fallback(ps)

load_parameters_fallback(ps) = ps

load_states(rng, model, st) = load_states_fallback(st)

function load_states_fallback(st)
    return fmap(st) do stᵢ
        stᵢ isa SerializedRNG && return Random.default_rng()
        return stᵢ
    end
end

end
