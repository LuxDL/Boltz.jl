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

function (tp::Layers.TensorProductLayer)(x::AnyTracedRArray{T,N}, ps, st) where {T,N}
    x′ = LuxOps.eachslice(x, Val(N - 1))                           # [I1, I2, ..., B] × T
    @argcheck length(x′) == length(tp.basis_fns)

    tmps = [
        eachslice(basis(x); dims=Tuple(2:(ndims(x) + 1))) for
        (basis, x) in zip(tp.basis_fns, x′)
    ]
    cur_val = vec(tmps[1])
    for i in 2:length(tmps)
        cur_val = Utils.safe_kron(cur_val, vec(tmps[i]))
    end

    z, stₙ = tp.dense(Utils.mapreduce_stack(cur_val), ps, st)
    return reshape(z, size(x)[1:(end - 2)]..., tp.out_dim, size(x)[end]), stₙ
end

end
