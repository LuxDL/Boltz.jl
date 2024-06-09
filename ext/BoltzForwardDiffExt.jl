module BoltzForwardDiffExt

using ADTypes: AutoForwardDiff
using Boltz: Boltz, Layers
using ForwardDiff: ForwardDiff

@inline Boltz._is_extension_loaded(::Val{:ForwardDiff}) = true

@inline Boltz._should_type_assert(::AbstractArray{<:ForwardDiff.Dual}) = false
@inline Boltz._should_type_assert(::ForwardDiff.Dual) = false

# Hamiltonian NN
function Layers.hamiltonian_forward(::AutoForwardDiff, model, x)
    return ForwardDiff.gradient(sum ∘ model, x)
end

end
