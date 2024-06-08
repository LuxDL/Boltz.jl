module BoltzZygoteExt

using ADTypes: AutoZygote
using Boltz: Boltz, Layers
using Zygote: Zygote

@inline Boltz._is_extension_loaded(::Val{:Zygote}) = true

# Hamiltonian NN
function Layers.hamiltonian_forward(::AutoZygote, model, x)
    return only(Zygote.gradient(sum ∘ model, x))
end

end
