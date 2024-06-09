@concrete struct KANDense{mode, N, PI <: NamedTuple, SI <: NamedTuple} <:
                 AbstractExplicitLayer
    input_dim::Int
    output_dim::Int
    degree::Int

    normalizer::N
    basis

    ps_init_fns::PI
    st_init_fns::SI
end

@generated function __init_from_namedtuple(
        rng::AbstractRNG, nt::NamedTuple{fields}, dims...) where {fields}
    syms = [gensym(Symbol(f)) for f in fields]
    calls = [:($(syms[i]) = nt.$(fields[i])(rng, dims...)) for i in eachindex(fields)]
    return quote
        $(calls...)
        return NamedTuple{$fields}(tuple($(syms...)))
    end
end

function LuxCore.initialparameters(rng::AbstractRNG, kan::KANDense)
    nt = kan.normalizer isa AbstractExplicitLayer ?
         (; ₋₋normalizer₋₋=LuxCore.initialparameters(rng, kan.normalizer)) : (;)
    return merge(
        __init_from_namedtuple(
            rng, kan.ps_init_fns, kan.degree, kan.input_dim, kan.output_dim),
        nt)
end

function LuxCore.initialstates(rng::AbstractRNG, kan::KANDense)
    nt = kan.normalizer isa AbstractExplicitLayer ?
         (; ₋₋normalizer₋₋=LuxCore.initialstates(rng, kan.normalizer)) : (;)
    return merge(
        __init_from_namedtuple(
            rng, kan.st_init_fns, kan.degree, kan.input_dim, kan.output_dim),
        nt)
end

@inline function __apply_normalizer(norm::AbstractExplicitLayer, x, ps, st)
    x_norm, st_norm = norm(x, ps.₋₋normalizer₋₋, st.₋₋normalizer₋₋)
    return x_norm, (; ₋₋normalizer₋₋=st_norm)
end

@inline __apply_normalizer(norm, x, ps, st) = norm(x), (;)

# From here we implement specific KAN variants

function KANDense(
        input_dim::Int, output_dim::Int, cheby::Basis.GeneralBasisFunction{:Chebyshev};
        normalizer=Base.BroadcastFunction(NNlib.tanh_fast), init_cheby_coeffs=randn32)
    return KANDense{:Chebyshev}(input_dim, output_dim, cheby.n, normalizer,
        cheby, (; cheby_coeffs=init_cheby_coeffs), (;))
end

function (kan::KANDense{:Chebyshev})(x::AbstractArray{T, N}, ps, st) where {T, N}
    @argcheck size(x, 1) == kan.input_dim && N ≥ 2

    x_norm, st_norm = __apply_normalizer(kan.normalizer, x, ps, st) # size(x)
    y = reshape(kan.basis(x_norm), kan.degree, kan.input_dim, 1, :) # D x I x 1 x B′
    z = dropdims(prod(y .* ps.cheby_coeffs; dims=(1, 2)); dims=(1, 2)) # D x O x B′

    return reshape(z, kan.output_dim, size(x)[2:end]...), merge(st, st_norm)
end
