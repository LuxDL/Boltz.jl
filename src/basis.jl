module Basis

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArgCheck: @argcheck
    using ..Boltz: Boltz, _unsqueeze1
    using ChainRulesCore: ChainRulesCore, NoTangent
    using ConcreteStructs: @concrete
    using LuxDeviceUtils: get_device, LuxCPUDevice
    using Markdown: @doc_str
    using NNlib: tanh_fast
end

const CRC = ChainRulesCore

abstract type AbstractBasisFunction end

@inline __basis_broadcast(f::F, i, x, ::Int) where {F} = f.(i, x)

@concrete struct SimpleBasisFunction{name} <: AbstractBasisFunction
    f
    n::Int
    dim::Int
end

function Base.show(io::IO, basis::SimpleBasisFunction{name}) where {name}
    print(io, "Basis.$(name)(order=$(basis.n))")
end

@inline function (basis::SimpleBasisFunction{name, F})(x::AbstractArray,
        grid::Union{AbstractRange, AbstractVector}=1:1:(basis.n)) where {name, F}
    @argcheck length(grid) == basis.n
    if basis.dim == 1 # Fast path where we don't need to materialize the range
        return __basis_broadcast(basis.f, grid, _unsqueeze1(x), 1)
    end

    @argcheck ndims(x) + 1 ≥ basis.dim
    new_x_size = ntuple(
        i -> i == basis.dim ? 1 : (i < basis.dim ? size(x, i) : size(x, i - 1)),
        ndims(x) + 1)
    x_new = reshape(x, new_x_size)
    if grid isa AbstractRange
        dev = get_device(x)
        grid = dev isa LuxCPUDevice ? collect(grid) : dev(grid)
    end
    grid_shape = ntuple(i -> i == basis.dim ? basis.n : 1, ndims(x) + 1)
    grid_new = reshape(grid, grid_shape)
    return __basis_broadcast(basis.f, grid_new, x_new, basis.dim)
end

@doc doc"""
    Chebyshev(n; dim::Int=1)

Constructs a Chebyshev basis of the form $[T_{0}(x), T_{1}(x), \dots, T_{n-1}(x)]$ where
$T_j(.)$ is the $j^{th}$ Chebyshev polynomial of the first kind.

## Arguments

  - `n`: number of terms in the polynomial expansion.

## Keyword Arguments

  - `dim::Int=1`: The dimension along which the basis functions are applied.
"""
Chebyshev(n; dim::Int=1) = SimpleBasisFunction{:Chebyshev}(__chebyshev, n, dim)

@inline __chebyshev(i, x) = @fastmath cos(i * acos(x))

@fastmath function CRC.rrule(
        ::typeof(__basis_broadcast), ::typeof(__chebyshev), i, x, dims::Int)
    iacosx = @. i * acos(x)
    y = @. cos(iacosx)

    ∇chebyshev = let iacosx = iacosx, i = i, x = x
        Δ -> begin
            den = @. sqrt(1 - x^2)
            return (NoTangent(), NoTangent(), NoTangent(),
                dropdims(sum(i .* sin.(iacosx) .* Δ ./ den; dims); dims))
        end
    end

    return y, ∇chebyshev
end

@doc doc"""
    Sin(n; dim::Int=1)

Constructs a sine basis of the form $[\sin(x), \sin(2x), \dots, \sin(nx)]$.

## Arguments

  - `n`: number of terms in the sine expansion.

## Keyword Arguments

  - `dim::Int=1`: The dimension along which the basis functions are applied.
"""
Sin(n; dim::Int=1) = SimpleBasisFunction{:Sin}(@fastmath(sin∘*), n, dim)

@doc doc"""
    Cos(n; dim::Int=1)

Constructs a cosine basis of the form $[\cos(x), \cos(2x), \dots, \cos(nx)]$.

## Arguments

  - `n`: number of terms in the cosine expansion.

## Keyword Arguments

  - `dim::Int=1`: The dimension along which the basis functions are applied.
"""
Cos(n; dim::Int=1) = SimpleBasisFunction{:Cos}(@fastmath(cos∘*), n, dim)

@doc doc"""
    Fourier(n; dim=1)

Constructs a Fourier basis of the form

$$F_j(x) = \begin{cases}
    cos\left(\frac{j}{2}x\right) & \text{if } j \text{ is even} \\
    sin\left(\frac{j}{2}x\right) & \text{if } j \text{ is odd}
\end{cases}$$

## Arguments

  - `n`: number of terms in the Fourier expansion.

## Keyword Arguments

  - `dim::Int=1`: The dimension along which the basis functions are applied.
"""
Fourier(n; dim::Int=1) = SimpleBasisFunction{:Fourier}(__fourier, n, dim)

@inline @fastmath function __fourier(i, x::AbstractFloat)
    s, c = sincos(i * x / 2)
    return ifelse(iseven(i), c, s)
end

@inline function __fourier(i, x) # No FastMath for non abstract floats
    s, c = sincos(i * x / 2)
    return ifelse(iseven(i), c, s)
end

@fastmath function CRC.rrule(
        ::typeof(__basis_broadcast), ::typeof(__fourier), i, x, dims::Int)
    ix_by_2 = @. i * x / 2
    s = @. sin(ix_by_2)
    c = @. cos(ix_by_2)
    y = @. ifelse(iseven(i), c, s)

    ∇fourier = let s = s, c = c, i = i
        Δ -> begin
            return (NoTangent(), NoTangent(), NoTangent(),
                dropdims(sum((i / 2) .* ifelse.(iseven.(i), -s, c) .* Δ; dims); dims))
        end
    end

    return y, ∇fourier
end

@doc doc"""
    Legendre(n; dim::Int=1)

Constructs a Legendre basis of the form $[P_{0}(x), P_{1}(x), \dots, P_{n-1}(x)]$ where
$P_j(.)$ is the $j^{th}$ Legendre polynomial.

## Arguments

  - `n`: number of terms in the polynomial expansion.

## Keyword Arguments

  - `dim::Int=1`: The dimension along which the basis functions are applied.
"""
Legendre(n; dim::Int=1) = SimpleBasisFunction{:Legendre}(__legendre_poly, n, dim)

## Source: https://github.com/ranocha/PolynomialBases.jl/blob/master/src/legendre.jl
@inline function __legendre_poly(i, x)
    p = i - 1
    a = one(x)
    b = x

    p ≤ 0 && return a
    p == 1 && return b

    for j in 2:p
        a, b = b, @fastmath(((2j - 1) * x * b - (j - 1) * a)/j)
    end

    return b
end

@doc doc"""
    Polynomial(n; dim::Int=1)

Constructs a Polynomial basis of the form $[1, x, \dots, x^{(n-1)}]$.

## Arguments

  - `n`: number of terms in the polynomial expansion.

## Keyword Arguments

  - `dim::Int=1`: The dimension along which the basis functions are applied.
"""
Polynomial(n; dim::Int=1) = SimpleBasisFunction{:Polynomial}(__polynomial, n, dim)

@inline __polynomial(i, x) = x^(i - 1)

function CRC.rrule(::typeof(__basis_broadcast), ::typeof(__polynomial), i, x, dims::Int)
    y_m1 = x .^ (i .- 2)
    y = y_m1 .* x
    ∇polynomial = let y_m1 = y_m1, i = i
        Δ -> begin
            return (NoTangent(), NoTangent(), NoTangent(),
                dropdims(sum((i .- 1) .* y_m1 .* Δ; dims); dims))
        end
    end
    return y, ∇polynomial
end

# Part of these are taken from https://github.com/vpuri3/KolmogorovArnold.jl/blob/master/src/utils.jl
@concrete struct RadialBasisFunction{name} <: AbstractBasisFunction
    f
    ϵ
    dim::Int
end

function Base.show(io::IO, basis::RadialBasisFunction{name}) where {name}
    print(io, "Basis.$(name)(ϵ=$(basis.ϵ))")
end

@inline function (basis::RadialBasisFunction{name, F})(
        x::AbstractArray, grid::AbstractVector) where {name, F}
    if basis.dim == 1 # Fast path where we don't need to materialize the range
        return basis.f((_unsqueeze1(x) .- grid) .* basis.ϵ)
    end

    @argcheck ndims(x) + 1 ≥ basis.dim
    new_x_size = ntuple(
        i -> i == basis.dim ? 1 : (i < basis.dim ? size(x, i) : size(x, i - 1)),
        ndims(x) + 1)
    x_new = reshape(x, new_x_size)
    grid_shape = ntuple(i -> i == basis.dim ? length(grid) : 1, ndims(x) + 1)
    grid_new = reshape(grid, grid_shape)
    return basis.f((x_new .- grid_new) .* basis.ϵ)
end

function GaussianRBF(ϵ; approx_exp::Bool=false, dim::Int=1)
    return RadialBasisFunction{:GaussianRBF}(Base.Fix2(__gaussian_rbf, approx_exp), ϵ, dim)
end

function InverseQuadraticRBF(ϵ; dim::Int=1)
    return RadialBasisFunction{:InverseQuadraticRBF}(__inverse_quadratic_rbf, ϵ, dim)
end

function InverseMultiquadicRBF(ϵ; dim::Int=1)
    return RadialBasisFunction{:InverseMultiquadicRBF}(__inverse_multiquadic_rbf, ϵ, dim)
end

RSWAF(ϵ; dim::Int=1) = RadialBasisFunction{:RSWAF}(__rswaf, ϵ, dim)

@inline function __gaussian_rbf(y, approx_exp::Bool)
    approx_exp && return @. Boltz.fast_approx_exp(-y^2)
    return @fastmath @. exp(-y^2)
end

@fastmath @inline function CRC.rrule(
        ::typeof(__gaussian_rbf), y::AbstractArray{T}, approx_exp::Bool) where {T}
    z = __gaussian_rbf(y, approx_exp)
    ∇gaussian_rbf = let y = y, z = z, T = T
        Δ -> (NoTangent(), -T(2) .* y .* z .* Δ)
    end
    return z, ∇gaussian_rbf
end

@inline __inverse_quadratic_rbf(y) = @fastmath @. 1 / (1 + y^2)

@inline __inverse_multiquadic_rbf(y) = @fastmath @. 1 / sqrt(1 + y^2)

@inline __rswaf(y) = @fastmath @. 1 - tanh_fast(y)^2

@fastmath @inline function CRC.rrule(::typeof(__rswaf), y::AbstractArray{T}) where {T}
    tx = @. tanh_fast(y)
    z = @. T(1) - tx^2
    ∇rswaf = let z = z, tx = tx, T = T
        Δ -> (NoTangent(), -T(2) .* tx .* z .* Δ)
    end
    return z, ∇rswaf
end

end
