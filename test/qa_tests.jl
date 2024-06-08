@testitem "Aqua: Quality Assurance" tags=[:others] begin
    using Aqua

    Aqua.test_all(Boltz)
end

@testitem "Explicit Imports: Quality Assurance" tags=[:others] begin
    # Load all trigger packages
    import Lux, Metalhead
    using ExplicitImports

    # Skip our own packages
    @test check_no_implicit_imports(Boltz; skip=(Boltz, Base, Core, Lux)) === nothing
    ## AbstractRNG seems to be a spurious detection in LuxFluxExt
    @test check_no_stale_explicit_imports(Boltz) === nothing
    @test check_all_qualified_accesses_via_owners(Boltz) === nothing
end
