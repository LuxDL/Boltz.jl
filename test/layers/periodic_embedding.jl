using Reactant, Boltz, Lux, StableRNGs, Test, LuxTestUtils

include("../testutils.jl")

dev = reactant_device(; force=true)

layer = Layers.PeriodicEmbedding([2, 3], [4.0, π / 5])
ps, st = Lux.setup(StableRNG(0), layer)
ps_ra, st_ra = (ps, st) |> dev

x = randn(StableRNG(0), Float32, 6, 4, 3, 2)
x_ra = x |> dev
Δx = Float32[0.0, 12.0, -2π / 5, 0.0, 0.0, 0.0]

# Test periodicity
val = Array(layer(x, ps, st)[1])
shifted_val = Array(layer(x .+ Δx, ps, st)[1])

@test all(val[1:4, :, :, :] .== shifted_val[1:4, :, :, :])
@test all(isapprox.(val[5:8, :, :, :], shifted_val[5:8, :, :, :]; atol=1e-5))

@test_gradients(
    TestUtils.sumabs2first,
    Constant(layer),
    x,
    ps,
    Constant(st);
    atol=1.0f-3,
    rtol=1.0f-3,
    broken_backends=[AutoEnzyme()]
)

st_test = Lux.testmode(st)
st_ra_test = st_test |> dev

@test @jit(layer(x_ra, ps_ra, st_ra_test))[1] ≈ layer(x, ps, st_test)[1] atol = 1e-3 rtol =
    1e-3

dx_ra, dps_ra = TestUtils.compute_reactant_gradient(layer, x_ra, ps_ra, st_ra)
dx_fd, dps_fd = TestUtils.compute_reactant_gradient_fd(layer, x_ra, ps_ra, st_ra)

@test dx_ra ≈ dx_fd atol = 1e-3 rtol = 1e-3
@test LuxTestUtils.check_approx(dps_ra, dps_fd; atol=1e-3, rtol=1e-3)
