@testsetup module SharedTestSetup

using Enzyme
Enzyme.API.runtimeActivity!(true)

import Reexport: @reexport
@reexport using Boltz, Lux, GPUArraysCore, LuxLib, LuxTestUtils, Random, StableRNGs,
                MLDataDevices
import Metalhead

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

export MODES, BACKEND_GROUP

end
