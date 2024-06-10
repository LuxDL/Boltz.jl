# TODO: We want to clean up the struct once we have the variants we want implemented
@concrete struct KANDense{mode, PI <: NamedTuple, SI <: NamedTuple} <: AbstractExplicitLayer
    input_dim::Int
    output_dim::Int

    basis

    ps_init_fns::PI
    st_init_fns::SI
end

@generated function __init_from_namedtuple(
        rng::AbstractRNG, nt::NamedTuple{fields}) where {fields}
    syms = [gensym(Symbol(f)) for f in fields]
    calls = [:($(syms[i]) = ((nt.$(fields[i]))[1])(rng, (nt.$(fields[i]))[2]...))
             for i in eachindex(fields)]
    return quote
        $(calls...)
        return NamedTuple{$fields}(tuple($(syms...)))
    end
end

function LuxCore.initialparameters(rng::AbstractRNG, kan::KANDense)
    return __init_from_namedtuple(rng, kan.ps_init_fns)
end

function LuxCore.initialstates(rng::AbstractRNG, kan::KANDense)
    return __init_from_namedtuple(rng, kan.st_init_fns)
end

# From here we implement specific KAN variants

"""
    KANDense(input_dim::Int, output_dim::Int, basis::Basis.SimpleBasisFunction;
        init_coefficients=randn32)

Constructs a Kolmogorov-Arnold Network (KAN) dense layer with `Basis.SimpleBasisFunction`.
A common example of this is Chebychev KAN which is constructed using `Basis.Chebyshev`.
"""
function KANDense(input_dim::Int, output_dim::Int,
        basis::Basis.SimpleBasisFunction; init_coefficients=randn32)
    return KANDense{:SimpleBasisFunction}(input_dim, output_dim, basis,
        (; coefficients=(init_coefficients, (basis.n, input_dim, output_dim))), (;))
end

function (kan::KANDense{:SimpleBasisFunction})(x::AbstractArray{T, N}, ps, st) where {T, N}
    @argcheck size(x, 1) == kan.input_dim && N ≥ 2
    x_size_rem = size(x)[2:end]

    y = reshape(kan.basis(x), :, kan.input_dim, 1, prod(x_size_rem))    # D x I x 1 x B′
    z = dropdims(prod(y .* ps.coefficients; dims=(1, 2)); dims=(1, 2))  # D x O x B′

    return reshape(z, kan.output_dim, x_size_rem...), st
end

"""
    KANDense(input_dim::Int, output_dim::Int, basis::Basis.AbstractRadialBasisFunction;
        grid_min=-1.0f0, grid_max=1.0f0, grid_size=8, init_weight=kaiming_normal,
        epsilon=0.1f0)

Constructs a Kolmogorov-Arnold Network (KAN) dense layer with
`Basis.AbstractRadialBasisFunction`. A common example of this is RBF KAN which is
constructed using `Basis.GaussianRBF` (the version used in `FastKAN`, though it is not
really the fastest version).
"""
function KANDense(input_dim::Int, output_dim::Int, basis::Basis.AbstractRadialBasisFunction;
        grid_min=-1.0f0, grid_max=1.0f0, grid_size::Int=8,
        init_weight=kaiming_normal, epsilon=0.1f0)
    init_grid = (rng, gmin, gmax, gsize) -> collect(LinRange(gmin, gmax, gsize))
    init_epsilon = (rng, eps) -> eps
    return KANDense{:RadialBasisFunction}(input_dim,
        output_dim,
        basis,
        (; weight=(init_weight, (output_dim, grid_size * input_dim))),
        (; grid=(init_grid, (grid_min, grid_max, grid_size)),
            epsilon=(init_epsilon, (epsilon,))))
end

function (kan::KANDense{:RadialBasisFunction})(x::AbstractArray{T, N}, ps, st) where {T, N}
    @argcheck size(x, 1) == kan.input_dim && N ≥ 2
    x_size_rem = size(x)[2:end]

    basis = kan.basis(x, st.grid, st.epsilon)   # G x I x ....
    y = reshape(basis, :, prod(x_size_rem))     # (G x I) x 1 x B′
    z = ps.weight * y                           # O x B′

    return reshape(z, kan.output_dim, x_size_rem...), st
end
