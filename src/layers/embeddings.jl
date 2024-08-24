"""
    ClassTokens(dim; init=zeros32)

Appends class tokens to an input with embedding dimension `dim` for use in many vision
transformer models.
"""
@concrete struct ClassTokens <: AbstractExplicitLayer
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
@concrete struct ViPosEmbedding <: AbstractExplicitLayer
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
