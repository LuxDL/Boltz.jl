module Basis

using ArgCheck: @argcheck
using ..Boltz: _unsqueeze1
using ChainRulesCore: ChainRulesCore, NoTangent
using ConcreteStructs: @concrete
using Markdown: @doc_str
using MLDataDevices: get_device, CPUDevice

const CRC = ChainRulesCore

# The rrules in this file are hardcoded to be used exclusively with GeneralBasisFunction
@concrete struct GeneralBasisFunction{name}
    f
    n::Int
    dim::Int
end

function Base.show(io::IO, basis::GeneralBasisFunction{name}) where {name}
    print(io, "Basis.$(name)(order=$(basis.n))")
end

@inline function (basis::GeneralBasisFunction{name, F})(x::AbstractArray,
        grid::Union{AbstractRange, AbstractVector}=1:1:(basis.n)) where {name, F}
    @argcheck length(grid) == basis.n
    if basis.dim == 1 # Fast path where we don't need to materialize the range
        return basis.f.(grid, _unsqueeze1(x))
    end

    @argcheck ndims(x) + 1 ≥ basis.dim
    new_x_size = ntuple(
        i -> i == basis.dim ? 1 : (i < basis.dim ? size(x, i) : size(x, i - 1)),
        ndims(x) + 1)
    x_new = reshape(x, new_x_size)
    if grid isa AbstractRange
        dev = get_device(x)
        grid = dev isa CPUDevice ? collect(grid) : dev(grid)
    end
    grid_shape = ntuple(i -> i == basis.dim ? basis.n : 1, ndims(x) + 1)
    grid_new = reshape(grid, grid_shape)
    return basis.f.(grid_new, x_new)
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
Chebyshev(n; dim::Int=1) = GeneralBasisFunction{:Chebyshev}(__chebyshev, n, dim)

@inline __chebyshev(i, x) = @fastmath cos(i * acos(x))

@doc doc"""
    Sin(n; dim::Int=1)

Constructs a sine basis of the form $[\sin(x), \sin(2x), \dots, \sin(nx)]$.

## Arguments

  - `n`: number of terms in the sine expansion.

## Keyword Arguments

$(DIM_KWARG_DOC)
"""
Sin(n; dim::Int=1) = GeneralBasisFunction{:Sin}(@fastmath(sin∘*), n, dim)

@doc doc"""
    Cos(n; dim::Int=1)

Constructs a cosine basis of the form $[\cos(x), \cos(2x), \dots, \cos(nx)]$.

## Arguments

  - `n`: number of terms in the cosine expansion.

## Keyword Arguments

$(DIM_KWARG_DOC)
"""
Cos(n; dim::Int=1) = GeneralBasisFunction{:Cos}(@fastmath(cos∘*), n, dim)

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
Fourier(n; dim::Int=1) = GeneralBasisFunction{:Fourier}(__fourier, n, dim)

@inline @fastmath function __fourier(i, x::AbstractFloat)
    s, c = sincos(i * x / 2)
    return ifelse(iseven(i), c, s)
end

@inline function __fourier(i, x) # No FastMath for non abstract floats
    s, c = sincos(i * x / 2)
    return ifelse(iseven(i), c, s)
end

@fastmath function CRC.rrule(::typeof(Broadcast.broadcasted), ::typeof(__fourier), i, x)
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
Legendre(n; dim::Int=1) = GeneralBasisFunction{:Legendre}(__legendre_poly, n, dim)

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
Polynomial(n; dim::Int=1) = GeneralBasisFunction{:Polynomial}(__polynomial, n, dim)

@inline __polynomial(i, x) = x^(i - 1)

function CRC.rrule(::typeof(Broadcast.broadcasted), ::typeof(__polynomial), i, x)
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

end
