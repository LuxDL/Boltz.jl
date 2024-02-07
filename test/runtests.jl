using SafeTestsets, Test, TestSetExtensions

const GROUP = get(ENV, "GROUP", "All")

@testset ExtendedTestSet "Boltz.jl" begin
    @safetestset "Vision Models" include("vision.jl")
end
