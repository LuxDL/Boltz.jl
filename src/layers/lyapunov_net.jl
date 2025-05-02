"""
    PositiveDefinite(model, x0; ψ, r)
    PositiveDefinite(model, x0; f1, f2, P, Q)
    PositiveDefinite(model; in_dims, ψ, r)
    PositiveDefinite(model; in_dims, f1, f2, P, Q)

Constructs a Lyapunov-Net [gaby2022lyapunovnetdeepneuralnetwork](@citep), which is positive
definite about `x0` whenever `ψ` and `r` meet certain conditions described below.

For a model `ϕ`,
`PositiveDefinite(ϕ, ψ, r, x0)(x, ps, st) = ψ(ϕ(x, ps, st) - ϕ(x0, ps, st)) + r(x, x0)`.
This results in a model which maps `x0` to `0` and any other input to a positive number
(i.e., a model which is positive definite about `x0`) whenever `ψ` is positive definite
about zero and `r` returns a positive number for any non-equal inputs and zero for equal
inputs.

A special constructor is provided for ``ψ(Δϕ) = f_1(Δϕ^* P Δϕ)`` and
``r(x, x_0) = f_2((x - x_0)^* Q (x - x_0))``, where ``P`` and ``Q`` are positive definite
matrices, and ``f_1`` and ``f_2`` map zero to zero and a positive number to a positive
number. If your `ψ` and `r` are of this form, using the corresponding constructor will
result in serious performance improvements. This optimized form is also used when no keyword
arguments (besides `in_dims`) are provided.

## Arguments

  - `model`: the underlying model being transformed into a positive definite function
  - `x0`: The unique input that will be mapped to zero instead of a positive number

## Keyword Arguments

  - `in_dims`: the number of input dimensions if `x0` is not provided; uses
    `x0 = zeros(in_dims)`
  - `ψ`: a positive definite function (about zero); used with `r` and defaults to
    ``ψ(x) = ||x||^2``
  - `r`: a bivariate function such that `r(x0, x0) = 0` and
    `r(x, x0) > 0` whenever `x ≠ x0`; used with `ψ` and defaults to
    ``r(x, y) = ||x - y||^2``
  - `f1`: a function that takes a scalar and returns a scalar; used with `f2`, `P`, and `Q`;
    defaults to `identity`
  - `f2`: a function that takes a scalar and returns a scalar; used with `f1`, `P`, and `Q`;
    defaults to `identity`
  - `P`: a positive definite matrix; used with `f1`, `f2`, and `Q`; defaults to `I`
  - `Q`: a positive definite matrix; used with `f1`, `f2`, and `P`; defaults to `I`

## Inputs

  - `x`: will be passed directly into `model`, so must meet the input requirements of that
    argument

## Returns

  - The output of the positive definite model
  - The state of the positive definite model. If the underlying model changes it state, the
    state will be updated first according to the call with the input `x0`, then according to
    the call with the input `x`.

## States

  - `st`: a `NamedTuple` containing the state of the underlying `model` and the `x0` value.
    When using the `f1`, `f2`, `P`, and `Q` constructor, the state will also contain the
    `P` and `Q` matrices.

## Parameters

  - Same as the underlying `model`
"""
@concrete struct PositiveDefinite <: AbstractLuxWrapperLayer{:model}
    model <: AbstractLuxLayer
    init_x0 <: Function
    in_dims::Integer
    ψ <: Function
    r <: Function

    function PositiveDefinite(model, x0::AbstractVector; kwargs...)
        if keys(kwargs) ⊆ [:f1, :f2, :P, :Q]
            f1 = get(kwargs, :f1, identity)
            f2 = get(kwargs, :f2, identity)
            P = get(kwargs, :P, I)
            Q = get(kwargs, :Q, I)
            ψ = f1 ∘ Base.Fix2(quadratic_form, copy(P))
            r = f2 ∘ Base.Fix2(quadratic_form, copy(Q)) ∘ -
        elseif keys(kwargs) ⊆ [:ψ, :r]
            ψ = get(kwargs, :ψ, identity ∘ Base.Fix2(quadratic_form, I))
            r = get(kwargs, :r, identity ∘ Base.Fix2(quadratic_form, I) ∘ -)
        else
            throw(
                ArgumentError(
                    "Invalid keyword arguments for PositiveDefinite. Got $(keys(kwargs)) " *
                    "and expected a subset of either [:f1, :f2, :P, :Q] or [:ψ, :r]",
                ),
            )
        end
        return PositiveDefinite(model, Returns(copy(x0)), length(x0), ψ, r)
    end
    function PositiveDefinite(model; kwargs...)
        if !haskey(kwargs, :in_dims)
            throw(
                ArgumentError(
                    "PositiveDefinite requires in_dims to be specified when x0 is not " *
                    "provided.",
                ),
            )
        end
        in_dims = kwargs[:in_dims]
        if keys(kwargs) ⊆ [:in_dims, :f1, :f2, :P, :Q]
            f1 = get(kwargs, :f1, identity)
            f2 = get(kwargs, :f2, identity)
            P = get(kwargs, :P, I)
            Q = get(kwargs, :Q, I)
            ψ = f1 ∘ Base.Fix2(quadratic_form, copy(P))
            r = f2 ∘ Base.Fix2(quadratic_form, copy(Q)) ∘ -
        elseif keys(kwargs) ⊆ [:in_dims, :ψ, :r]
            ψ = get(kwargs, :ψ, identity ∘ Base.Fix2(quadratic_form, I))
            r = get(kwargs, :r, identity ∘ Base.Fix2(quadratic_form, I) ∘ -)
        else
            throw(
                ArgumentError(
                    "Invalid keyword arguments for PositiveDefinite. Got $(keys(kwargs)) " *
                    "and expected a subset of either [:in_dims, :f1, :f2, :P, :Q] or " *
                    "[:in_dims, :ψ, :r]",
                ),
            )
        end
        return PositiveDefinite(model, zeros32, in_dims, ψ, r)
    end
end

quadratic_form(x, Q) = x' * Q * x
quadratic_form(x, ::typeof(I)) = sum(abs2, x)

function LuxCore.initialstates(
    rng::AbstractRNG, pd::PositiveDefinite{L,F,Ψ,R}
) where {L,F,Ψ,R}
    return (; model=LuxCore.initialstates(rng, pd.model), x0=pd.init_x0(rng, pd.in_dims))
end

function (pd::PositiveDefinite{L,F,Ψ,R})(x::V, ps, st) where {V<:AbstractVector,L,F,Ψ,R}
    ϕ0, new_model_st = pd.model(st.x0, ps, st.model)
    ϕx, final_model_st = pd.model(x, ps, new_model_st)
    return V([pd.ψ(ϕx - ϕ0) + pd.r(x, st.x0)]), merge(st, (; model=final_model_st))
end

function (pd::PositiveDefinite{L,F,Ψ,R})(x::AbstractMatrix, ps, st) where {L,F,Ψ,R}
    ϕ0, new_model_st = pd.model(st.x0, ps, st.model)
    ϕx, final_model_st = pd.model(x, ps, new_model_st)
    init = @ignore_derivatives permutedims(empty(ϕ0))
    return (
        mapreduce(hcat, zip(eachcol(x), eachcol(ϕx)); init=init) do (x, ϕx)
            pd.ψ(ϕx - ϕ0) + pd.r(x, st.x0)
        end,
        merge(st, (; model=final_model_st)),
    )
end

const QuadraticLyapunovNet{L,F,F1,F2,P,Q} = PositiveDefinite{
    L,
    F,
    ComposedFunction{F1,Base.Fix2{typeof(quadratic_form),P}},
    ComposedFunction{ComposedFunction{F2,Base.Fix2{typeof(quadratic_form),Q}},typeof(-)},
} where {L,F,F1,F2,P,Q}

function LuxCore.initialstates(rng::AbstractRNG, qln::QuadraticLyapunovNet)
    return (;
        model=LuxCore.initialstates(rng, qln.model),
        x0=qln.init_x0(rng, qln.in_dims),
        P=copy(qln.ψ.inner.x),
        Q=copy(qln.r.outer.inner.x),
    )
end

function (qln::QuadraticLyapunovNet{L,F,F1,F2,P,Q})(
    x::V, ps, st
) where {V<:AbstractVector,L,F,F1,F2,P,Q}
    ϕ0, new_model_st = qln.model(st.x0, ps, st.model)
    ϕx, final_model_st = qln.model(x, ps, new_model_st)

    Δϕ = ϕx .- ϕ0
    ΔϕPΔϕ = quadratic_form(Δϕ, st.P)

    Δx = x .- st.x0
    ΔxQΔx = quadratic_form(Δx, st.Q)

    f1 = qln.ψ.outer
    f2 = qln.r.outer.outer

    return V([f1(ΔϕPΔϕ) + f2(ΔxQΔx)]), merge(st, (; model=final_model_st))
end

function (qln::QuadraticLyapunovNet{L,F,F1,F2,P,Q})(
    x::AbstractMatrix, ps, st
) where {L,F,F1,F2,P,Q}
    ϕ0, new_model_st = qln.model(st.x0, ps, st.model)
    ϕx, final_model_st = qln.model(x, ps, new_model_st)

    Δϕ = ϕx .- ϕ0
    PΔϕ = st.P * Δϕ
    ΔϕPΔϕ = sum(conj(Δϕ) .* PΔϕ; dims=1)

    Δx = x .- st.x0
    QΔx = st.Q * Δx
    ΔxQΔx = sum(conj(Δx) .* QΔx; dims=1)

    f1 = qln.ψ.outer
    f2 = qln.r.outer.outer

    return f1.(ΔϕPΔϕ) .+ f2.(ΔxQΔx), merge(st, (; model=final_model_st))
end

"""
    ShiftTo(model, in_val, out_val)

Vertically shifts the output of `model` to output `out_val` when the input is `in_val`.

For a model `ϕ`, `ShiftTo(ϕ, in_val, out_val)(x, ps, st) = ϕ(x, ps, st) + Δϕ`,
where `Δϕ = out_val - ϕ(in_val, ps, st)`.

## Arguments

  - `model`: the underlying model being transformed into a positive definite function
  - `in_val`: The input that will be mapped to `out_val`
  - `out_val`: The value that the output will be shifted to when the input is `in_val`

## Inputs

  - `x`: will be passed directly into `model`, so must meet the input requirements of that
    argument

## Returns

  - The output of the shifted model
  - The state of the shifted model. If the underlying model changes it state, the
    state will be updated first according to the call with the input `in_val`, then
    according to the call with the input `x`.

## States

  - `st`: a `NamedTuple` containing the state of the underlying `model` and the `in_val` and
    `out_val` values

## Parameters

  - Same as the underlying `model`
"""
@concrete struct ShiftTo <: AbstractLuxWrapperLayer{:model}
    model <: AbstractLuxLayer
    init_in_val <: Function
    init_out_val <: Function

    function ShiftTo(model, in_val::AbstractVector, out_val::AbstractVector)
        return ShiftTo(model, Returns(copy(in_val)), Returns(copy(out_val)))
    end
end

function LuxCore.initialstates(rng::AbstractRNG, s::ShiftTo)
    return (;
        model=LuxCore.initialstates(rng, s.model),
        in_val=s.init_in_val(),
        out_val=s.init_out_val(),
    )
end

function (s::ShiftTo)(x::AbstractVector, ps, st)
    out, new_st = s(reshape(x, :, 1), ps, st)
    return vec(out), new_st
end

function (s::ShiftTo)(x::AbstractMatrix, ps, st)
    ϕ0, new_model_st = s.model(st.in_val, ps, st.model)
    Δϕ = st.out_val .- ϕ0
    ϕx, final_model_st = s.model(x, ps, new_model_st)
    return (ϕx .+ Δϕ, merge(st, (; model=final_model_st)))
end
