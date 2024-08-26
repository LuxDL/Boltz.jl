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
julia> layer = PeriodicEmbedding([2], [4.0])
PeriodicEmbedding([2], [4.0])

julia> using Random;
       rng = Random.seed!(123);

julia> ps, st = Lux.setup(rng, layer)
(NamedTuple(), (k = [0.5],))

julia> all(layer([1.1, 2.2, 3.3], ps, st)[1] .==
           [1.1, 3.3, sinpi(2 / 4.0 * 2.2), cospi(2 / 4.0 * 2.2)])
true
```
"""
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
    other_idxs = @ignore_derivatives setdiff(axes(x, 1), st.idxs)
    return (
        vcat(x[other_idxs, :], sinpi.(st.k .* x[p.idxs, :]), cospi.(st.k .* x[p.idxs, :])),
        st)
end

function (p::PeriodicEmbedding)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(first(p(reshape(x, size(x, 1), :), ps, st)), :, size(x)[2:end]...), st
end

function Base.show(io::IO, ::MIME"text/plain", p::PeriodicEmbedding)
    return print(io, "PeriodicEmbedding(", p.idxs, ", ", p.periods, ")")
end
