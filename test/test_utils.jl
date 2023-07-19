using Boltz, Lux, LuxCUDA, LuxLib, LuxTestUtils

const GROUP = get(ENV, "GROUP", "All")

@static if VERSION ≥ v"1.9"
    if GROUP == "All" || GROUP == "AMDGPU"
        using LuxAMDGPU
    end
end

CUDA.allowscalar(false)

cpu_testing() = GROUP == "All" || GROUP == "CPU"
cuda_testing() = (GROUP == "All" || GROUP == "CUDA") && LuxCUDA.functional()
function amdgpu_testing()
    @static if VERSION ≥ v"1.9"
        return (GROUP == "All" || GROUP == "AMDGPU") && LuxAMDGPU.functional()
    else
        return false
    end
end

const MODES = begin
    # Mode, Array Type, Device Function, GPU?
    cpu_mode = ("CPU", Array, LuxCPUDevice(), false)
    cuda_mode = ("CUDA", CuArray, LuxCUDADevice(), true)
    amdgpu_mode = @static if VERSION ≥ v"1.9"
        if GROUP == "All" || GROUP == "AMDGPU"
            ("AMDGPU", ROCArray, LuxAMDGPUDevice(), true)
        else
            nothing
        end
    else
        nothing
    end

    modes = []
    cpu_testing() && push!(modes, cpu_mode)
    cuda_testing() && push!(modes, cuda_mode)
    amdgpu_testing() && push!(modes, amdgpu_mode)

    modes
end
