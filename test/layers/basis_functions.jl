using Reactant, Boltz, Lux, StableRNGs, Test

include("../testutils.jl")

dev = reactant_device(; force=true)

# Test for each basis type
@testset "$(basis)" for basis in (
    Basis.Chebyshev,
    Basis.Sin,
    Basis.Cos,
    Basis.Fourier,
    Basis.Polynomial,
    # Basis.Legendre # TODO: tracing
)
    x = tanh.(randn(Float32, 2, 4))
    x_ra = x |> dev
    grid = collect(Float32, 1:3)
    grid_ra = grid |> dev

    # Test with default dimension (dim=1)
    fn1 = basis(3)
    @test size(fn1(x)) == (3, 2, 4)
    @test size(fn1(x, grid)) == (3, 2, 4)

    # Test with dim=2
    fn2 = basis(3; dim=2)
    @test size(fn2(x)) == (2, 3, 4)
    @test size(fn2(x, grid)) == (2, 3, 4)

    # Test with dim=3
    fn3 = basis(3; dim=3)
    @test size(fn3(x)) == (2, 4, 3)
    @test size(fn3(x, grid)) == (2, 4, 3)

    # Test dimension error
    fn4 = basis(3; dim=4)
    @test_throws ArgumentError fn4(x)

    grid2 = collect(Float32, 1:5)
    @test_throws ArgumentError fn4(x, grid2)

    # Test Reactant compilation
    @test @jit(fn1(x_ra)) ≈ fn1(x) atol = 1e-3 rtol = 1e-3
    @test @jit(fn1(x_ra, grid_ra)) ≈ fn1(x, grid) atol = 1e-3 rtol = 1e-3

    @test @jit(fn2(x_ra)) ≈ fn2(x) atol = 1e-3 rtol = 1e-3
    @test @jit(fn2(x_ra, grid_ra)) ≈ fn2(x, grid) atol = 1e-3 rtol = 1e-3

    @test @jit(fn3(x_ra)) ≈ fn3(x) atol = 1e-3 rtol = 1e-3
    @test @jit(fn3(x_ra, grid_ra)) ≈ fn3(x, grid) atol = 1e-3 rtol = 1e-3
end
