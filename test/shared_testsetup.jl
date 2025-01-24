@testsetup module SharedTestSetup

import Reexport: @reexport
@reexport using Boltz, Lux, GPUArraysCore, LuxLib, LuxTestUtils, Random, StableRNGs,
                Reactant
using MLDataDevices, JLD2, Enzyme, Zygote
using LuxTestUtils: check_approx

LuxTestUtils.jet_target_modules!(["Boltz", "Lux", "LuxLib"])

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))

GPUArraysCore.allowscalar(false)

if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    using LuxCUDA
end

if BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu"
    using AMDGPU
end

cpu_testing() = BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
function cuda_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") &&
           MLDataDevices.functional(CUDADevice)
end
function amdgpu_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") &&
           MLDataDevices.functional(AMDGPUDevice)
end

const MODES = begin
    modes = []
    cpu_testing() && push!(modes, ("cpu", Array, CPUDevice(), false))
    cuda_testing() && push!(modes, ("cuda", CuArray, CUDADevice(), true))
    amdgpu_testing() && push!(modes, ("amdgpu", ROCArray, AMDGPUDevice(), true))
    modes
end

test_reactant(mode::String) = mode != "amdgpu"
function set_reactant_backend!(mode::String)
    if mode == "cuda"
        Reactant.set_default_backend("gpu")
    elseif mode == "cpu"
        Reactant.set_default_backend("cpu")
    else
        error("Unknown mode $(mode)")
    end
end

sumabs2first(model, x, ps, st) = sum(abs2, first(model(x, ps, st)))

function compute_reactant_gradient(model, x, ps, st)
    return compute_reactant_gradient(sumabs2first, model, x, ps, st)
end

function compute_reactant_gradient(f::F, model, x, ps, st) where {F}
    res = Enzyme.gradient(Reverse, f, Const(model), x, ps, Const(st))
    return res[2], res[3]
end

function compute_zygote_gradient(model, x, ps, st)
    return compute_zygote_gradient(sumabs2first, model, x, ps, st)
end

function compute_zygote_gradient(f::F, model, x, ps, st) where {F}
    return Zygote.gradient((x, ps) -> f(model, x, ps, st), x, ps)
end

export MODES, BACKEND_GROUP, test_reactant, set_reactant_backend!,
       compute_reactant_gradient, compute_zygote_gradient, check_approx

end
