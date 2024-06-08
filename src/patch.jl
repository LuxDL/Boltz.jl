# Hacky Patch for loading pretrained models
@static if VERSION ≥ v"1.10-"
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
