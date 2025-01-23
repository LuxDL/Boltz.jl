"""
    ClassTokens(dim; init=zeros32)

Appends class tokens to an input with embedding dimension `dim` for use in many vision
transformer models.
"""
@concrete struct ClassTokens <: AbstractLuxLayer
    dim::Int
    init
end

ClassTokens(dim::Int; init=zeros32) = ClassTokens(dim, init)

LuxCore.initialparameters(rng::AbstractRNG, c::ClassTokens) = (; token=c.init(rng, c.dim))

function (m::ClassTokens)(x::AbstractArray{T, N}, ps, st) where {T, N}
    tokens = reshape(ps.token, :, ntuple(_ -> 1, N - 1)...) .* ones_batch_like(x)
    return cat(x, tokens; dims=Val(N - 1)), st
end

function ones_batch_like(x::AbstractArray{T, N}) where {T, N}
    return fill!(similar(x, ntuple(_ -> 1, N - 1)..., size(x, N)), one(T))
end

@non_differentiable ones_batch_like(x::AbstractArray)

"""
    ViPosEmbedding(embedding_size, number_patches; init = randn32)

Positional embedding layer used by many vision transformer-like models.
"""
@concrete struct ViPosEmbedding <: AbstractLuxLayer
    embedding_size::Int
    number_patches::Int
    init
end

function ViPosEmbedding(embedding_size::Int, number_patches::Int; init=randn32)
    return ViPosEmbedding(embedding_size, number_patches, init)
end

function LuxCore.initialparameters(rng::AbstractRNG, v::ViPosEmbedding)
    return (; vectors=v.init(rng, v.embedding_size, v.number_patches))
end

(v::ViPosEmbedding)(x, ps, st) = x .+ ps.vectors, st

"""
    PeriodicEmbedding(idxs, periods)

Create an embedding periodic in some inputs with specified periods. Input indices not in
`idxs` are passed through unchanged, but inputs in `idxs` are moved to the end of the
output and replaced with their sines, followed by their cosines (scaled appropriately to
have the specified periods). This smooth embedding preserves phase information and enforces
periodicity.

For example, `layer = PeriodicEmbedding([2, 3], [3.0, 1.0])` will create a layer periodic in
the second input with period 3.0 and periodic in the third input with period 1.0. In this
case, `layer([a, b, c, d], st) == ([a, d, sinpi(2 / 3.0 * b), sinpi(2 / 1.0 * c), cospi(2 / 3.0 * b), cospi(2 / 1.0 * c)], st)`.

## Arguments

  - `idxs`: Indices of the periodic inputs
  - `periods`: Periods of the periodic inputs, in the same order as in `idxs`

## Inputs

  - `x` must be an `AbstractArray` with `issubset(idxs, axes(x, 1))`
  - `st` must be a `NamedTuple` where `st.k = 2 ./ periods`, but on the same device as `x`

## Returns

  - `AbstractArray` of size `(size(x, 1) + length(idxs), ...)` where `...` are the other
    dimensions of `x`.
  - `st`, unchanged

## Example

```jldoctest
julia> layer = Layers.PeriodicEmbedding([2], [4.0])
PeriodicEmbedding([2], [4.0])

julia> ps, st = Lux.setup(Random.default_rng(), layer);

julia> all(layer([1.1, 2.2, 3.3], ps, st)[1] .==
           [1.1, 3.3, sinpi(2 / 4.0 * 2.2), cospi(2 / 4.0 * 2.2)])
true
```
"""
@concrete struct PeriodicEmbedding <: AbstractLuxLayer
    idxs
    periods
end

function LuxCore.initialstates(::AbstractRNG, pe::PeriodicEmbedding)
    return (; idxs=DataTransferBarrier(pe.idxs), k=2 ./ pe.periods)
end

function (pe::PeriodicEmbedding)(x::AbstractVector, ps, st::NamedTuple)
    return vec(first(pe(reshape(x, :, 1), ps, st))), st
end

function (p::PeriodicEmbedding)(x::AbstractMatrix, _, st::NamedTuple)
    idxs = st.idxs.val
    other_idxs = @ignore_derivatives @allowscalar setdiff(axes(x, 1), idxs)
    y = vcat(x[other_idxs, :], sinpi.(st.k .* x[idxs, :]), cospi.(st.k .* x[idxs, :]))
    return y, st
end

function (p::PeriodicEmbedding)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(first(p(reshape(x, size(x, 1), :), ps, st)), :, size(x)[2:end]...), st
end

function Base.show(io::IO, ::MIME"text/plain", p::PeriodicEmbedding)
    return print(io, "PeriodicEmbedding(", p.idxs, ", ", p.periods, ")")
end

"""
    PatchEmbedding(image_size, patch_size, in_channels, embed_planes;
        norm_layer=Returns(Lux.NoOpLayer()), flatten=true)

Constructs a patch embedding layer with the given image size, patch size, input channels,
and embedding planes. The patch size must be a divisor of the image size.

## Arguments

  - `image_size`: image size as a tuple
  - `patch_size`: patch size as a tuple
  - `in_channels`: number of input channels
  - `embed_planes`: number of embedding planes

## Keyword Arguments

  - `norm_layer`: Takes the embedding planes as input and returns a layer that normalizes
    the embedding planes. Defaults to no normalization.
  - `flatten`: set to `true` to flatten the output of the convolutional layer
"""
@concrete struct PatchEmbedding <: AbstractLuxContainerLayer{(:patch, :flatten, :norm)}
    patch <: AbstractLuxLayer
    flatten <: AbstractLuxLayer
    norm <: AbstractLuxLayer
end

function (pe::PatchEmbedding)(x, ps, st)
    y₁, st₁ = pe.patch(x, ps.patch, st.patch)
    y₂, st₂ = pe.flatten(y₁, ps.flatten, st.flatten)
    y₃, st₃ = pe.norm(y₂, ps.norm, st.norm)
    return y₃, (; patch=st₁, flatten=st₂, norm=st₃)
end

function PatchEmbedding(image_size::Dims{N}, patch_size::Dims{N}, in_channels::Int,
        embed_planes::Int; norm_layer=Returns(Lux.NoOpLayer()), flatten::Bool=true) where {N}
    foreach(zip(image_size, patch_size)) do (i, p)
        @argcheck i % p==0 "Image size ($i) must be divisible by patch size ($p)"
    end

    return PatchEmbedding(
        Lux.Conv(patch_size, in_channels => embed_planes; stride=patch_size),
        ifelse(flatten, Lux.WrappedFunction(flatten_spatial), Lux.NoOpLayer()),
        norm_layer(embed_planes)
    )
end
