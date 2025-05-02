module BoltzReactantExt

using ADTypes: AutoEnzyme
using Reactant: Reactant, AnyTracedRArray

using Boltz: Boltz, Layers, Utils

Utils.is_extension_loaded(::Val{:Reactant}) = true

Layers.get_hamiltonian_autodiff(::AutoEnzyme, ::AnyTracedRArray) = AutoEnzyme()
function Layers.get_hamiltonian_autodiff(autodiff, ::AnyTracedRArray)
    @warn "`autodiff` = `$autodiff` is not supported with Reactant. Falling back to \
           `AutoEnzyme`." maxlog = 1
    return AutoEnzyme()
end

end
