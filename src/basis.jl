module Basis

using ..Boltz: _unsqueeze1
using ChainRulesCore: ChainRulesCore, NoTangent
using ConcreteStructs: @concrete
using Markdown: @doc_str

const CRC = ChainRulesCore

# The rrules in this file are hardcoded to be used exclusively with GeneralBasisFunction
@concrete struct GeneralBasisFunction{name}
    f
    n::Int
end

function Base.show(io::IO, basis::GeneralBasisFunction{name}) where {name}
    print(io, "Basis.$(name)(order=$(basis.n))")
end

@inline function (basis::GeneralBasisFunction{name, F})(x::AbstractArray) where {name, F}
    return basis.f.(1:(basis.n), _unsqueeze1(x))
end

@doc doc"""
    Chebyshev(n)

Constructs a Chebyshev basis of the form $[T_{0}(x), T_{1}(x), \dots, T_{n-1}(x)]$ where
$T_j(.)$ is the $j^{th}$ Chebyshev polynomial of the first kind.

## Arguments

  - `n`: number of terms in the polynomial expansion.
"""
Chebyshev(n) = GeneralBasisFunction{:Chebyshev}(__chebyshev, n)

@inline __chebyshev(i, x) = @fastmath cos(i * acos(x))

@doc doc"""
    Sin(n)

Constructs a sine basis of the form $[\sin(x), \sin(2x), \dots, \sin(nx)]$.

## Arguments

  - `n`: number of terms in the sine expansion.
"""
Sin(n) = GeneralBasisFunction{:Sin}(@fastmath(sin∘*), n)

@doc doc"""
    Cos(n)

Constructs a cosine basis of the form $[\cos(x), \cos(2x), \dots, \cos(nx)]$.

## Arguments

  - `n`: number of terms in the cosine expansion.
"""
Cos(n) = GeneralBasisFunction{:Cos}(@fastmath(cos∘*), n)

@doc doc"""
    Fourier(n)

Constructs a Fourier basis of the form
$F_j(x) = j is even ? cos((j÷2)x) : sin((j÷2)x)$ => $[F_0(x), F_1(x), \dots, F_n(x)]$.

## Arguments

  - `n`: number of terms in the Fourier expansion.
"""
Fourier(n) = GeneralBasisFunction{:Fourier}(__fourier, n)

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
    Legendre(n)

Constructs a Legendre basis of the form $[P_{0}(x), P_{1}(x), \dots, P_{n-1}(x)]$ where
$P_j(.)$ is the $j^{th}$ Legendre polynomial.

## Arguments

  - `n`: number of terms in the polynomial expansion.
"""
Legendre(n) = GeneralBasisFunction{:Legendre}(__legendre_poly, n)

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
    Polynomial(n)

Constructs a Polynomial basis of the form $[1, x, \dots, x^(n-1)]$.

## Arguments

  - `n`: number of terms in the polynomial expansion.
"""
Polynomial(n) = GeneralBasisFunction{:Polynomial}(__polynomial, n)

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
