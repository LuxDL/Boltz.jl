using Boltz, Lux, LuxCUDA, LuxLib, LuxTestUtils

const GROUP = get(ENV, "GROUP", "All")

cpu_testing() = GROUP == "All" || GROUP == "CPU"
cuda_testing() = (GROUP == "All" || GROUP == "CUDA") && LuxCUDA.functional()
amdgpu_testing() = (GROUP == "All" || GROUP == "AMDGPU") # && LuxAMDGPU.functional()

const MODES = begin
    # Mode, Array Type, GPU?
    cpu_mode = ("CPU", cpu, false)
    cuda_mode = ("CUDA", gpu, true)

    modes = []
    cpu_testing() && push!(modes, cpu_mode)
    cuda_testing() && push!(modes, cuda_mode)
    modes
end
