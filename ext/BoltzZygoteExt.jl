module BoltzZygoteExt

using ADTypes: AutoZygote
using Zygote: Zygote

using Boltz: Boltz, Layers, Utils

Utils.is_extension_loaded(::Val{:Zygote}) = true

# Hamiltonian NN
function Layers.hamiltonian_forward(::AutoZygote, model, x)
    return only(Zygote.gradient(sum ∘ model, x))
end

end
