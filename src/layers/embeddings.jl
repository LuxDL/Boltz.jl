"""
    ClassTokens(dim; init=zeros32)

Appends class tokens to an input with embedding dimension `dim` for use in many vision
transformer models.
"""
@kwdef @concrete struct ClassTokens <: AbstractExplicitLayer
    dim::Int
    init = zeros32
end

@inline ClassTokens(dim::Int; init=zeros32) = ClassTokens(dim, init)

LuxCore.initialparameters(rng::AbstractRNG, c::ClassTokens) = (; token=c.init(rng, c.dim))

function (m::ClassTokens)(x::AbstractArray{T, N}, ps, st) where {T, N}
    tokens = reshape(ps.token, :, ntuple(_ -> 1, N - 1)...) .* __ones_like(x)
    return cat(x, tokens; dims=Val(N - 1)), st
end

@inline __ones_like(x::AbstractArray{T, N}) where {T, N} = fill!(
    similar(x, ntuple(_ -> 1, N - 1)..., size(x, N)), one(T))

CRC.@non_differentiable __ones_like(x::AbstractArray)

"""
    ViPosEmbedding(embedding_size, number_patches; init = randn32)

Positional embedding layer used by many vision transformer-like models.
"""
@kwdef @concrete struct ViPosEmbedding <: AbstractExplicitLayer
    embedding_size::Int
    number_patches::Int
    init = randn32
end

@inline ViPosEmbedding(embedding_size::Int, number_patches::Int; init=randn32) = ViPosEmbedding(
    embedding_size, number_patches, init)

function LuxCore.initialparameters(rng::AbstractRNG, v::ViPosEmbedding)
    return (; vectors=v.init(rng, v.embedding_size, v.number_patches))
end

@inline (v::ViPosEmbedding)(x, ps, st) = x .+ ps.vectors, st
