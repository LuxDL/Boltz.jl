module BoltzPickleExt

using Pickle: Pickle

using Boltz: InitializeModels, Utils

Utils.is_extension_loaded(::Val{:Pickle}) = true

function InitializeModels.load_using_pickle_internal(path, args...; kwargs...)
    ext = splitext(path)[2]
    # drop args here since they are generally not needed
    ext == ".pth" && return Pickle.Torch.THload(path; kwargs...)
    return error("Unknown pretrained format: $(ext)")
end

end
