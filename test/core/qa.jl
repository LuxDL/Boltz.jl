using Lux, Aqua, ExplicitImports, Boltz, Test

@testset "Aqua: Quality Assurance" begin
    Aqua.test_all(Boltz; ambiguities=false)
    Aqua.test_ambiguities(Boltz; recursive=false)
end

@testset "Explicit Imports: Quality Assurance" begin
    @test check_no_implicit_imports(Boltz; skip=(Base, Core, Lux)) === nothing
    @test check_no_stale_explicit_imports(Boltz) === nothing
    @test check_no_self_qualified_accesses(Boltz) === nothing
    @test check_all_explicit_imports_via_owners(Boltz) === nothing
    @test check_all_qualified_accesses_via_owners(Boltz) === nothing
    @test_broken check_all_explicit_imports_are_public(Boltz) === nothing  # mostly upstream problems
    @test_broken check_all_qualified_accesses_are_public(Boltz) === nothing  # mostly upstream
end
