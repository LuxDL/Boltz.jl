"""
    PositiveDefinite(model, x0; ψ, r)
    PositiveDefinite(model; in_dims, ψ, r)

Constructs a Lyapunov-Net [gaby2022lyapunovnetdeepneuralnetwork](@citep), which is positive
definite about `x0` whenever `ψ` and `r` meet certain conditions described below.

For a model `ϕ`,
`PositiveDefinite(ϕ, ψ, r, x0)(x, ps, st) = ψ(ϕ(x, ps, st) - ϕ(x0, ps, st)) + r(x, x0)`.
This results in a model which maps `x0` to `0` and any other input to a positive number
(i.e., a model which is positive definite about `x0`) whenever `ψ` is positive definite
about zero and `r` returns a positive number for any non-equal inputs and zero for equal
inputs.

## Arguments
  - `model`: the underlying model being transformed into a positive definite function
  - `x0`: The unique input that will be mapped to zero instead of a positive number

## Keyword Arguments
  - `in_dims`: the number of input dimensions if `x0` is not provided; uses
    `x0 = zeros(in_dims)`
  - `ψ`: a positive definite function (about zero); defaults to ``ψ(x) = ||x||^2``
  - `r`: a bivariate function such that `r(x0, x0) = 0` and
    `r(x, x0) > 0` whenever `x ≠ x0`; defaults to ``r(x, y) = ||x - y||^2``

## Inputs
  - `x`: will be passed directly into `model`, so must meet the input requirements of that
    argument

## Returns
  - The output of the positive definite model
  - The state of the positive definite model. If the underlying model changes it state, the
    state will be updated according to the call with the input `x`, not with the call using
    `x0`.

## States
  - `st`: a `NamedTuple` containing the state of the underlying `model` and the `x0` value

## Parameters
  - Same as the underlying `model`
"""
@concrete struct PositiveDefinite <: AbstractLuxWrapperLayer{:model}
    model <: AbstractLuxLayer
    init_x0 <: Function
    in_dims::Integer
    ψ <: Function
    r <: Function

    function PositiveDefinite(model, x0::AbstractVector; ψ=Base.Fix1(sum, abs2),
        r=Base.Fix1(sum, abs2) ∘ -)
        return PositiveDefinite(model, (rng, in_dims) -> copy(x0), length(x0), ψ, r)
    end
    function PositiveDefinite(model; in_dims::Integer, ψ=Base.Fix1(sum, abs2),
        r=Base.Fix1(sum, abs2) ∘ -)
        return PositiveDefinite(model, zeros32, in_dims, ψ, r)
    end
end

function LuxCore.initialstates(rng::AbstractRNG, pd::PositiveDefinite)
    return (; model=LuxCore.initialstates(rng, pd.model), x0=pd.init_x0(rng, pd.in_dims))
end

function (pd::PositiveDefinite)(x::AbstractVector, ps, st)
    out, new_st = pd(reshape(x, :, 1), ps, st)
    return vec(out), new_st
end

function (pd::PositiveDefinite)(x::AbstractMatrix, ps, st)
    ϕ0, _ = pd.model(st.x0, ps, st.model)
    ϕx, new_model_st = pd.model(x, ps, st.model)
    ϕx_cols = eachcol(ϕx)
    return (
        permutedims(
            mapreduce(vcat, zip(eachcol(x), ϕx_cols); init=empty(first(ϕx_cols))) do (x, ϕx)
                pd.ψ(ϕx - ϕ0) + pd.r(x, st.x0)
            end
        ),
        merge(st, (; model=new_model_st))
    )
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
    state will be updated according to the call with the input `x`, not the call using
    `in_val`.

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
        _in_val = copy(in_val)
        _out_val = copy(out_val)
        return ShiftTo(model, () -> _in_val, () -> _out_val)
    end
end

function LuxCore.initialstates(rng::AbstractRNG, s::ShiftTo)
    return (;
        model=LuxCore.initialstates(rng, s.model),
        in_val=s.init_in_val(),
        out_val=s.init_out_val()
    )
end

function (s::ShiftTo)(x::AbstractVector, ps, st)
    out, new_st = s(reshape(x, :, 1), ps, st)
    return vec(out), new_st
end

function (s::ShiftTo)(x::AbstractMatrix, ps, st)
    ϕ0, _ = s.model(st.in_val, ps, st.model)
    Δϕ = st.out_val .- ϕ0
    ϕx, new_model_st = s.model(x, ps, st.model)
    return (
        ϕx .+ Δϕ,
        merge(st, (; model=new_model_st))
    )
end
