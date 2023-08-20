using SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "All")

@static if VERSION ≥ v"1.9"
    using Pkg
    (GROUP == "All" || GROUP == "AMDGPU") && Pkg.add("LuxAMDGPU")
end

@time begin
    @testset "Boltz.jl" begin
        @time @safetestset "Vision Models" begin
            include("vision.jl")
        end
    end
end
