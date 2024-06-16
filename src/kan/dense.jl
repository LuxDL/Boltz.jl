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
        init_weight=kaiming_normal, use_bias=true, init_bias=zeros32)

Constructs a Kolmogorov-Arnold Network (KAN) dense layer with `Basis.SimpleBasisFunction`.
A common example of this is Chebychev KAN which is constructed using `Basis.Chebyshev`.
"""
function KANDense(input_dim::Int, output_dim::Int, basis::Basis.SimpleBasisFunction;
        init_weight=kaiming_normal, use_bias=true,
        init_bias=zeros32, light_version::Bool=false)
    @argcheck basis.dim == 1 + light_version

    init_grid = ((rng, basis) -> 1:(basis.n), (basis,))

    ps_init_fns = (;
        weight=(init_weight, (output_dim, ifelse(light_version, 1, basis.n) * input_dim)),)
    if use_bias
        ps_init_fns = merge(ps_init_fns, (; bias=(init_bias, (output_dim,))))
    end

    return KANDense{:SimpleBasisFunction}(
        input_dim, output_dim, basis, ps_init_fns, (; grid=init_grid))
end

"""
    KANDense(input_dim::Int, output_dim::Int, basis::Basis.RadialBasisFunction;
        grid_min=-1.0f0, grid_max=1.0f0, grid_size=8, init_weight=kaiming_normal,
        use_bias=true, init_bias=zeros32)

Constructs a Kolmogorov-Arnold Network (KAN) dense layer with
`Basis.AbstractRadialBasisFunction`. A common example of this is RBF KAN which is
constructed using `Basis.GaussianRBF` (the version used in `FastKAN`, though it is not
really the fastest version).
"""
function KANDense(input_dim::Int, output_dim::Int, basis::Basis.RadialBasisFunction;
        grid_min=-1.0f0, grid_max=1.0f0, grid_size::Int=8, light_version::Bool=false,
        init_weight=kaiming_normal, use_bias=true, init_bias=zeros32)
    @argcheck grid_min < grid_max && grid_size > 0
    @argcheck basis.dim == 1 + light_version

    init_grid = (rng, gmin, gmax, gsize) -> collect(LinRange(gmin, gmax, gsize))

    ps_init_fns = (;
        weight=(init_weight, (output_dim, ifelse(light_version, 1, grid_size) * input_dim)),)
    if use_bias
        ps_init_fns = merge(ps_init_fns, (; bias=(init_bias, (output_dim,))))
    end

    return KANDense{:RadialBasisFunction}(input_dim, output_dim, basis, ps_init_fns,
        (; grid=(init_grid, (grid_min, grid_max, grid_size))))
end

function (kan::KANDense)(x::AbstractVector{T}, ps, st) where {T}
    y, st = kan(reshape(x, :, 1), ps, st)
    return vec(y), st
end

function (kan::KANDense)(x::AbstractArray{T, N}, ps, st) where {T, N}
    @argcheck size(x, 1) == kan.input_dim
    x_size_rem = size(x)[2:end]

    basis = kan.basis(x, st.grid)

    if kan.basis.dim == 1       # Regular version
        y = reshape(basis, :, prod(x_size_rem))                          # (G x I) x B′
        z = fused_dense_bias_activation(
            identity, ps.weight, y, Lux._getproperty(ps, Val(:bias)))    # O x B′I
    elseif kan.basis.dim == 2   # Light version
        y = reshape(basis, size(basis, 1), :)                           # G x (I x B′)
        z_inter = fused_dense_bias_activation(
            identity, ps.weight, y, Lux._getproperty(ps, Val(:bias)))    # O x (I x B′)
        z = Boltz._seconddimmean(reshape(z_inter, kan.output_dim, length(st.grid), :))  # O x B′
    else
        throw(ArgumentError(lazy"Invalid basis dimension: $(kan.basis.dim)"))
    end

    return reshape(z, kan.output_dim, x_size_rem...), st
end
