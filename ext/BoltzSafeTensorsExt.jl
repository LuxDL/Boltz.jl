module BoltzSafeTensorsExt

using SafeTensors: SafeTensors, load_safetensors

using Boltz: InitializeModels, Utils

Utils.is_extension_loaded(::Val{:SafeTensors}) = true

function InitializeModels.load_using_safetensors_internal(path, args...; kwargs...)
    return load_safetensors(path, args...; kwargs...)
end

end
