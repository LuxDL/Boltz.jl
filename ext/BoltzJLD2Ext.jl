module BoltzJLD2Ext

using JLD2: JLD2
using Random: Random

using Boltz: InitializeModels, Utils

Utils.is_extension_loaded(::Val{:JLD2}) = true

function InitializeModels.load_using_jld2_internal(args...; kwargs...)
    return JLD2.load(args...; kwargs...)
end

function Base.convert(::Type{Random.Xoshiro},
        x::JLD2.ReconstructedStatic{
            Symbol("Random.Xoshiro"), (:s0, :s1, :s2, :s3), NTuple{4, UInt64}})
    return Random.Xoshiro(x.s0, x.s1, x.s2, x.s3)
end
function Base.convert(::Type{Random.Xoshiro},
        x::JLD2.ReconstructedStatic{:Xoshiro, (:s0, :s1, :s2, :s3), NTuple{4, UInt64}})
    return Random.Xoshiro(x.s0, x.s1, x.s2, x.s3)
end

end
