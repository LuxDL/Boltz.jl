"""
    _fast_chunk(x::AbstractArray, ::Val{n}, ::Val{dim})

Type-stable and faster version of `MLUtils.chunk`.
"""
@inline _fast_chunk(h::Int, n::Int) = (1:h) .+ h * (n - 1)
@inline function _fast_chunk(x::AbstractArray, h::Int, n::Int, ::Val{dim}) where {dim}
    return selectdim(x, dim, _fast_chunk(h, n))
end
@inline function _fast_chunk(x::AbstractArray, ::Val{N}, d::Val{D}) where {N, D}
    return _fast_chunk.((x,), size(x, D) ÷ N, 1:N, d)
end
@inline function _fast_chunk(
        x::GPUArraysCore.AnyGPUArray, h::Int, n::Int, ::Val{dim}) where {dim}
    return copy(selectdim(x, dim, _fast_chunk(h, n)))
end

"""
    _flatten_spatial(x::AbstractArray{T, 4})

Flattens the first 2 dimensions of `x`, and permutes the remaining dimensions to (2, 1, 3)
"""
@inline function _flatten_spatial(x::AbstractArray{T, 4}) where {T}
    return permutedims(reshape(x, (:, size(x, 3), size(x, 4))), (2, 1, 3))
end

"""
    _seconddimmean(x)

Computes the mean of `x` along dimension `2`
"""
@inline _seconddimmean(x) = dropdims(mean(x; dims=2); dims=2)

"""
    _should_type_assert(x)

In certain cases, to ensure type-stability we want to add type-asserts. But this won't work
for exotic types like `ForwardDiff.Dual`. We use this function to check if we should add a
type-assert for `x`.
"""
@inline _should_type_assert(::AbstractArray{T}) where {T} = isbitstype(T)
@inline _should_type_assert(::AbstractArray{<:ForwardDiff.Dual}) = false
@inline _should_type_assert(::ForwardDiff.Dual) = false
@inline _should_type_assert(x) = true

@inline _unsqueeze1(x::AbstractArray) = reshape(x, 1, size(x)...)
@inline _unsqueezeN(x::AbstractArray) = reshape(x, size(x)..., 1)

@inline _catN(x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N} = cat(
    x, y; dims=Val(N))

@inline _stack(xs) = mapreduce(_unsqueezeN, _catN, xs)
