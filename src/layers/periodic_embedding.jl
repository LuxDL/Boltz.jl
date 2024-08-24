@concrete struct PeriodicEmbedding <: AbstractExplicitLayer
    idxs
    periods
end

function LuxCore.initialstates(::AbstractRNG, pe::PeriodicEmbedding)
    return (; idxs=pe.idxs, k=2 ./ pe.periods)
end

function (pe::PeriodicEmbedding)(x::AbstractVector, ps, st)
    return vec(first(pe(reshape(x, :, 1), ps, st))), st
end

function (p::PeriodicEmbedding)(x::AbstractMatrix, _, st::NamedTuple)
    other_idxs = @ignore_derivatives setdiff(axes(x, 1), p.idxs)
    return (
        vcat(x[other_idxs, :], sinpi.(st.k .* x[p.idxs, :]), cospi.(st.k .* x[p.idxs, :])),
        st)
end

function (p::PeriodicEmbedding)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(first(p(reshape(x, size(x, 1), :), ps, st)), :, size(x)[2:end]...), st
end

function Base.show(io::IO, p::PeriodicEmbedding)
    return print(io, "PeriodicEmbedding(", p.idxs, ", ", p.periods, ")")
end
