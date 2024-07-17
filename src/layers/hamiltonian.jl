"""
    HamiltonianNN{FST}(model; autodiff=nothing) where {FST}

Constructs a Hamiltonian Neural Network [1]. This neural network is useful for learning
symmetries and conservation laws by supervision on the gradients of the trajectories. It
takes as input a concatenated vector of length `2n` containing the position (of size `n`)
and momentum (of size `n`) of the particles. It then returns the time derivatives for
position and momentum.

## Arguments

  - `FST`: If `true`, then the type of the state returned by the model must be same as the
    type of the input state. See the documentation on [`StatefulLuxLayer`](@ref) for more
    information.
  - `model`: A `Lux.AbstractExplicitLayer` neural network that returns the Hamiltonian of
    the system. The `model` must return a "batched scalar", i.e. all the dimensions of the
    output except the last one must be equal to 1. The last dimension must be equal to the
    batchsize of the input.

## Keyword Arguments

  - `autodiff`: The autodiff framework to be used for the internal Hamiltonian computation.
    The default is `nothing`, which selects the best possible backend available. The
    available options are `AutoForwardDiff` and `AutoZygote`.

## Autodiff Backends

| `autodiff`        | Package Needed   | Notes                                                                        |
|:----------------- |:---------------- |:---------------------------------------------------------------------------- |
| `AutoZygote`      | `Zygote.jl`      | Preferred Backend. Chosen if `Zygote` is loaded and `autodiff` is `nothing`. |
| `AutoForwardDiff` | `ForwardDiff.jl` | Chosen if `Zygote` is not loaded and `autodiff` is `nothing`.                |

!!! note

    This layer uses nested autodiff. Please refer to the manual entry on
    [Nested Autodiff](@ref nested_autodiff) for more information and known limitations.

## References

[1] Greydanus, Samuel, Misko Dzamba, and Jason Yosinski. "Hamiltonian Neural Networks."
Advances in Neural Information Processing Systems 32 (2019): 15379-15389.
"""
@concrete struct HamiltonianNN{FST} <: AbstractExplicitContainerLayer{(:model,)}
    model
    autodiff
end

function HamiltonianNN{FST}(model; autodiff=nothing) where {FST}
    if autodiff === nothing # Select best possible backend
        autodiff = ifelse(
            Boltz._is_extension_loaded(Val(:Zygote)), AutoZygote(), AutoForwardDiff())
    elseif autodiff isa AutoZygote
        autodiff = Boltz._is_extension_loaded(Val(:Zygote)) ? autodiff : nothing
    elseif !(autodiff isa AutoForwardDiff)
        throw(ArgumentError("Invalid autodiff backend: $(autodiff). Available options: \
                             `AutoForwardDiff`, or `AutoZygote`."))
    end
    return HamiltonianNN{FST}(model, autodiff)
end

function LuxCore.initialstates(rng::AbstractRNG, hnn::HamiltonianNN)
    return (; model=LuxCore.initialstates(rng, hnn.model), first_call=true)
end

hamiltonian_forward(::AutoForwardDiff, model, x) = ForwardDiff.gradient(sum ∘ model, x)

function (hnn::HamiltonianNN{FST})(x::AbstractArray{T, N}, ps, st) where {FST, T, N}
    model = StatefulLuxLayer{FST}(hnn.model, ps, st.model)

    st.first_call && __check_hamiltonian_layer(hnn.model, x, ps, st.model)

    if _should_type_assert(x) && _should_type_assert(ps)
        H = hamiltonian_forward(hnn.autodiff, model, x)::typeof(x)
    else
        H = hamiltonian_forward(hnn.autodiff, model, x)
    end
    n = size(H, N - 1) ÷ 2
    return (
        cat(selectdim(H, N - 1, (n + 1):(2n)), selectdim(H, N - 1, 1:n); dims=Val(N - 1)),
        (; model=model.st, first_call=false))
end

function __check_hamiltonian_layer(model, x::AbstractArray{T, N}, ps, st) where {T, N}
    y = first(model(x, ps, st))
    _size = size(y)[1:(end - 1)]
    @argcheck all(isone, _size) && size(y, ndims(y)) == size(x, N)
end

CRC.@non_differentiable __check_hamiltonian_layer(::Any...)
