module Basis

using ArgCheck: @argcheck
using ..Boltz: _unsqueeze1
using ChainRulesCore: ChainRulesCore, NoTangent
using ConcreteStructs: @concrete
using LuxDeviceUtils: get_device, LuxCPUDevice
using Markdown: @doc_str

const CRC = ChainRulesCore

abstract type AbstractBasisFunction end

@inline __basis_broadcast(f::F, i, x) where {F} = f.(i, x)

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
        return __basis_broadcast(basis.f, grid, _unsqueeze1(x))
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
    return __basis_broadcast(basis.f, grid_new, x_new)
end

const DIM_KWARG_DOC = "  - `dim::Int=1`: The dimension along which the basis functions are applied."

@doc doc"""
    Chebyshev(n; dim::Int=1)

Constructs a Chebyshev basis of the form $[T_{0}(x), T_{1}(x), \dots, T_{n-1}(x)]$ where
$T_j(.)$ is the $j^{th}$ Chebyshev polynomial of the first kind.

## Arguments

  - `n`: number of terms in the polynomial expansion.

## Keyword Arguments

$(DIM_KWARG_DOC)
"""
Chebyshev(n; dim::Int=1) = SimpleBasisFunction{:Chebyshev}(__chebyshev, n, dim)

@inline __chebyshev(i, x) = @fastmath cos(i * acos(x))

@fastmath function CRC.rrule(::typeof(__basis_broadcast), ::typeof(__chebyshev), i, x)
    iacosx = @. i * acos(x)
    y = @. cos(iacosx)

    ∇chebyshev = let iacosx = iacosx, i = i, x = x
        Δ -> begin
            den = @. sqrt(1 - x^2)
            return (NoTangent(), NoTangent(), NoTangent(),
                dropdims(sum(i .* sin.(iacosx) .* Δ ./ den; dims=1); dims=1))
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

$(DIM_KWARG_DOC)
"""
Sin(n; dim::Int=1) = SimpleBasisFunction{:Sin}(@fastmath(sin∘*), n, dim)

@doc doc"""
    Cos(n; dim::Int=1)

Constructs a cosine basis of the form $[\cos(x), \cos(2x), \dots, \cos(nx)]$.

## Arguments

  - `n`: number of terms in the cosine expansion.

## Keyword Arguments

$(DIM_KWARG_DOC)
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

$(DIM_KWARG_DOC)
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

@fastmath function CRC.rrule(::typeof(__basis_broadcast), ::typeof(__fourier), i, x)
    ix_by_2 = @. i * x / 2
    s = @. sin(ix_by_2)
    c = @. cos(ix_by_2)
    y = @. ifelse(iseven(i), c, s)

    ∇fourier = let s = s, c = c, i = i
        Δ -> begin
            return (NoTangent(), NoTangent(), NoTangent(),
                dropdims(sum((i / 2) .* ifelse.(iseven.(i), -s, c) .* Δ; dims=1); dims=1))
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

$(DIM_KWARG_DOC)
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

$(DIM_KWARG_DOC)
"""
Polynomial(n; dim::Int=1) = SimpleBasisFunction{:Polynomial}(__polynomial, n, dim)

@inline __polynomial(i, x) = x^(i - 1)

function CRC.rrule(::typeof(__basis_broadcast), ::typeof(__polynomial), i, x)
    y_m1 = x .^ (i .- 2)
    y = y_m1 .* x
    ∇polynomial = let y_m1 = y_m1, i = i
        Δ -> begin
            return (NoTangent(), NoTangent(), NoTangent(),
                dropdims(sum((i .- 1) .* y_m1 .* Δ; dims=1); dims=1))
        end
    end
    return y, ∇polynomial
end

abstract type AbstractRadialBasisFunction <: AbstractBasisFunction end

# Part of these are taken from https://github.com/vpuri3/KolmogorovArnold.jl/blob/master/src/utils.jl
struct GaussianRBF <: AbstractRadialBasisFunction end

@fastmath @inline function (::GaussianRBF)(x, grid::AbstractVector, ϵ)
    return __gaussian_rbf((_unsqueeze1(x) .- grid) .* ϵ)
end

@inline __gaussian_rbf(y) = @fastmath @. exp(-y^2)

@fastmath @inline function CRC.rrule(
        ::typeof(__gaussian_rbf), y::AbstractArray{T}) where {T}
    z = __gaussian_rbf(y)
    ∇gaussian_rbf = let y = y, z = z, T = T
        Δ -> (NoTangent(), -T(2) .* y .* z .* Δ)
    end
    return z, ∇gaussian_rbf
end

struct InverseQuadraticRBF <: AbstractRadialBasisFunction end

@fastmath @inline function (::InverseQuadraticRBF)(x, grid::AbstractVector, ϵ)
    z = ((_unsqueeze1(x) .- grid) .* ϵ) .^ 2
    return 1 ./ (1 .+ z)
end

struct InverseMultiquadicRBF <: AbstractRadialBasisFunction end

@fastmath @inline function (::InverseMultiquadicRBF)(x, grid::AbstractVector, ϵ)
    return sqrt.(InverseQuadraticRBF()(x, grid, ϵ))
end

end
