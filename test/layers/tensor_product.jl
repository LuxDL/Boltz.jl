using Reactant, Boltz, Lux, StableRNGs, Test, LuxTestUtils

include("../testutils.jl")

dev = reactant_device(; force=true)

# Test for each basis type
@testset "$(basis)" for basis in (
    Basis.Chebyshev,
    Basis.Sin,
    Basis.Cos,
    Basis.Fourier,
    # Basis.Legendre, # TODO: tracing
    Basis.Polynomial,
)
    tensor_project = Layers.TensorProductLayer([basis(n + 2) for n in 1:3], 4)
    ps, st = Lux.setup(StableRNG(0), tensor_project)
    ps_ra, st_ra = (ps, st) |> dev

    # Test dimension mismatch error
    x_bad = tanh.(randn(Float32, 2, 4, 5))
    @test_throws ArgumentError tensor_project(x_bad, ps, st)

    x = tanh.(randn(Float32, 2, 3, 5))
    x_ra = x |> dev

    y, st = tensor_project(x, ps, st)
    @test size(y) == (2, 4, 5)

    @test_gradients(
        TestUtils.sumabs2first,
        Constant(tensor_project),
        x,
        ps,
        Constant(st);
        atol=1e-3,
        rtol=1e-3,
        skip_backends=[AutoEnzyme()]
    )

    st_test = Lux.testmode(st)
    st_ra_test = st_test |> dev

    @test @jit(tensor_project(x_ra, ps_ra, st_ra_test))[1] ≈
        tensor_project(x, ps, st_test)[1] atol = 1e-3 rtol = 1e-3

    dx_ra, dps_ra = TestUtils.compute_reactant_gradient(tensor_project, x_ra, ps_ra, st_ra)
    dx_fd, dps_fd = TestUtils.compute_reactant_gradient_fd(
        tensor_project, x_ra, ps_ra, st_ra
    )

    @test dx_ra ≈ dx_fd atol = 1e-3 rtol = 1e-3
    @test LuxTestUtils.check_approx(dps_ra, dps_fd; atol=1e-3, rtol=1e-3)
end
