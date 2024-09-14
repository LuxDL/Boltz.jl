module BoltzJLD2Ext

using JLD2: JLD2

using Boltz: InitializeModels, Utils

Utils.is_extension_loaded(::Val{:JLD2}) = true

function InitializeModels.load_using_jld2_internal(args...; kwargs...)
    return JLD2.load(args...; kwargs...)
end

end
