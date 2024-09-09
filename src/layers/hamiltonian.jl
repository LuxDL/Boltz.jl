"""
    HamiltonianNN{FST}(model; autodiff=nothing) where {FST}

Constructs a Hamiltonian Neural Network [greydanus2019hamiltonian](@citep). This neural
network is useful for learning symmetries and conservation laws by supervision on the
gradients of the trajectories. It takes as input a concatenated vector of length `2n`
containing the position (of size `n`) and momentum (of size `n`) of the particles. It then
returns the time derivatives for position and momentum.

## Arguments

  - `FST`: If `true`, then the type of the state returned by the model must be same as the
    type of the input state. See the documentation on `StatefulLuxLayer` for more
    information.
  - `model`: A `Lux.AbstractLuxLayer` neural network that returns the Hamiltonian of
    the system. The `model` must return a "batched scalar", i.e. all the dimensions of the
    output except the last one must be equal to 1. The last dimension must be equal to the
    batchsize of the input.

## Keyword Arguments

  - `autodiff`: The autodiff framework to be used for the internal Hamiltonian computation.
    The default is `nothing`, which selects the best possible backend available. The
    available options are `AutoForwardDiff` and `AutoZygote`.

## Autodiff Backends

| `autodiff`        | Package Needed | Notes                                                                        |
|:----------------- |:-------------- |:---------------------------------------------------------------------------- |
| `AutoZygote`      | `Zygote.jl`    | Preferred Backend. Chosen if `Zygote` is loaded and `autodiff` is `nothing`. |
| `AutoForwardDiff` |                | Chosen if `Zygote` is not loaded and `autodiff` is `nothing`.                |

!!! note

    This layer uses nested autodiff. Please refer to the manual entry on
    [Nested Autodiff](https://lux.csail.mit.edu/stable/manual/nested_autodiff) for more
    information and known limitations.
"""
@concrete struct HamiltonianNN{FST} <: AbstractLuxWrapperLayer{:model}
    model
    autodiff
end

function HamiltonianNN{FST}(model; autodiff=nothing) where {FST}
    @argcheck autodiff isa Union{Nothing, AutoForwardDiff, AutoZygote}

    zygote_loaded = is_extension_loaded(Val(:Zygote))

    if autodiff === nothing # Select best possible backend
        autodiff = ifelse(zygote_loaded, AutoZygote(), AutoForwardDiff())
    else
        if autodiff isa AutoZygote && !zygote_loaded
            throw(ArgumentError("`autodiff` cannot be `AutoZygote` when `Zygote.jl` is not \
                                 loaded."))
        end
    end

    return HamiltonianNN{FST}(model, autodiff)
end

function LuxCore.initialstates(rng::AbstractRNG, hnn::HamiltonianNN)
    return (; model=LuxCore.initialstates(rng, hnn.model), first_call=true)
end

hamiltonian_forward(::AutoForwardDiff, model, x) = ForwardDiff.gradient(sum ∘ model, x)

function (hnn::HamiltonianNN)(x::AbstractVector, ps, st)
    y, stₙ = hnn(reshape(x, :, 1), ps, st)
    return vec(y), stₙ
end

function (hnn::HamiltonianNN{FST})(x::AbstractArray{T, N}, ps, st) where {FST, T, N}
    model = StatefulLuxLayer{FST}(hnn.model, ps, st.model)

    st.first_call && check_hamiltonian_layer(hnn.model, x, ps, st.model)

    if should_type_assert(x) && should_type_assert(ps)
        H = hamiltonian_forward(hnn.autodiff, model, x)::typeof(x)
    else
        H = hamiltonian_forward(hnn.autodiff, model, x)
    end
    n = size(H, N - 1) ÷ 2
    return (
        cat(selectdim(H, N - 1, (n + 1):(2n)), selectdim(H, N - 1, 1:n); dims=Val(N - 1)),
        (; model=model.st, first_call=false))
end

function check_hamiltonian_layer(model, x::AbstractArray{T, N}, ps, st) where {T, N}
    y = first(model(x, ps, st))
    @argcheck all(isone, size(y)[1:(end - 1)]) && size(y, ndims(y)) == size(x, N)
end

@non_differentiable check_hamiltonian_layer(::Any...)
