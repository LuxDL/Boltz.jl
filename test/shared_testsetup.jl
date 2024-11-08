@testsetup module SharedTestSetup

import Reexport: @reexport
@reexport using Boltz, Lux, GPUArraysCore, LuxLib, LuxTestUtils, Random, StableRNGs
using MLDataDevices, JLD2

LuxTestUtils.jet_target_modules!(["Boltz", "Lux", "LuxLib"])

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))

GPUArraysCore.allowscalar(false)

if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    using LuxCUDA
end

if BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu"
    using AMDGPU
end

@static if !Sys.iswindows()
    @reexport using Reactant
    test_reactant(mode) = mode != "amdgpu"
    function set_reactant_backend!(mode)
        if mode == "cuda"
            Reactant.set_default_backend("gpu")
        elseif mode == "cpu"
            Reactant.set_default_backend("cpu")
        end
    end
else
    test_reactant(::Any) = true
    set_reactant_backend!(::Any) = nothing
    macro compile(expr)
        return :()
    end
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

export MODES, BACKEND_GROUP
export test_reactant, set_reactant_backend!
export @compile

end
