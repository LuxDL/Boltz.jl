module BoltzReactantExt

using ADTypes: AutoEnzyme
using Reactant: Reactant, AnyTracedRArray, TracedRNumber, Ops, @trace

using Boltz: Boltz, Layers, Basis, Utils

Utils.is_extension_loaded(::Val{:Reactant}) = true

Layers.get_hamiltonian_autodiff(::AutoEnzyme, ::AnyTracedRArray) = AutoEnzyme()
function Layers.get_hamiltonian_autodiff(autodiff, ::AnyTracedRArray)
    @warn "`autodiff` = `$autodiff` is not supported with Reactant. Falling back to \
           `AutoEnzyme`." maxlog = 1
    return AutoEnzyme()
end

function Utils.unsqueeze1(x::AnyTracedRArray)
    return Ops.reshape(Reactant.TracedUtils.materialize_traced_array(x), [1, size(x)...])
end
function Utils.unsqueezeN(x::AnyTracedRArray)
    return Ops.reshape(Reactant.TracedUtils.materialize_traced_array(x), [size(x)..., 1])
end

end
