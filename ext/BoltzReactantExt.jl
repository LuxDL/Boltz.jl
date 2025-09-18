module BoltzReactantExt

using ArgCheck: @argcheck
using ADTypes: AutoEnzyme
using Reactant: Reactant, AnyTracedRArray, AnyTracedRVector, TracedRNumber, Ops, @opcall
using ReactantCore: materialize_traced_array, @trace

using Lux: LuxOps
using Boltz: Boltz, Layers, Utils, Basis

Utils.is_extension_loaded(::Val{:Reactant}) = true

Layers.get_hamiltonian_autodiff(::AutoEnzyme, ::AnyTracedRArray) = AutoEnzyme()
function Layers.get_hamiltonian_autodiff(autodiff, ::AnyTracedRArray)
    @warn "`autodiff` = `$autodiff` is not supported with Reactant. Falling back to \
           `AutoEnzyme`." maxlog = 1
    return AutoEnzyme()
end

function Utils.unsqueeze1(x::AnyTracedRArray)
    return @opcall reshape(materialize_traced_array(x), [1, size(x)...])
end
function Utils.unsqueezeN(x::AnyTracedRArray)
    return @opcall reshape(materialize_traced_array(x), [size(x)..., 1])
end

end
