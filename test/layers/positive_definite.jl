using Reactant, Boltz, Lux, StableRNGs, Test, LuxTestUtils

include("../testutils.jl")

dev = reactant_device(; force=true)

model = Layers.MLP(2, (4, 4, 2), gelu)
pd = Layers.PositiveDefinite(model; in_dims=2)
ps, st = Lux.setup(StableRNG(0), pd)
ps_ra, st_ra = (ps, st) |> dev

x = randn(StableRNG(0), Float32, 2, 2)
x_ra = x |> dev
x0 = zeros(Float32, 2)

# Verify positive definite property by hand
y, _ = pd(x, ps, st)
z, _ = model(x, ps, st.model)
z0, _ = model(x0, ps, st.model)
y_by_hand = sum(abs2, z .- z0; dims=1) .+ sum(abs2, x .- x0; dims=1)

@test maximum(abs, y - y_by_hand) < 1.0f-8

@test_gradients(
    TestUtils.sumabs2first,
    Constant(pd),
    x,
    ps,
    Constant(st);
    atol=1.0f-3,
    rtol=1.0f-3,
    broken_backends=[AutoEnzyme()]
)

# Test with explicit reference point
pd2 = Layers.PositiveDefinite(model, ones(2))
ps2, st2 = Lux.setup(StableRNG(0), pd2)

x0_ones = ones(Float32, 2)
y2, _ = pd2(x0_ones, ps2, st2)

@test maximum(abs, y2) < 1.0f-8

# Test Reactant
st_test = Lux.testmode(st)
st_ra_test = st_test |> dev

@test @jit(pd(x_ra, ps_ra, st_ra_test))[1] ≈ pd(x, ps, st_test)[1] atol = 1e-3 rtol = 1e-3

dx_ra, dps_ra = TestUtils.compute_reactant_gradient(pd, x_ra, ps_ra, st_ra)
dx_fd, dps_fd = TestUtils.compute_reactant_gradient_fd(pd, x_ra, ps_ra, st_ra)

@test dx_ra ≈ dx_fd atol = 1e-3 rtol = 1e-3
@test LuxTestUtils.check_approx(dps_ra, dps_fd; atol=1e-3, rtol=1e-3)
