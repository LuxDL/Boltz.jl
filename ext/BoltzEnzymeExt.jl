module BoltzEnzymeExt

using ADTypes: AutoEnzyme
using Enzyme: Enzyme

using Boltz: Boltz, Layers, Utils

Utils.is_extension_loaded(::Val{:Enzyme}) = true

# Hamiltonian NN
function Layers.hamiltonian_forward(::AutoEnzyme, model, x)
    return only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(sum ∘ model), x))
end

end
