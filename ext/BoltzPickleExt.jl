module BoltzPickleExt

using Pickle: Pickle

using Boltz: InitializeModels, Utils

Utils.is_extension_loaded(::Val{:Pickle}) = true

function InitializeModels.load_using_pickle_internal(path, args...; kwargs...)
    ext = splitext(path)[2]
    ext == ".pth" && return Pickle.Torch.THload(path, args...; kwargs...)
    return error("Unknown pretrained format: $(ext)")
end

end
