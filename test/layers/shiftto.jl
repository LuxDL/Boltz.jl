using Reactant, Boltz, Lux, StableRNGs, Test, LuxTestUtils

include("../testutils.jl")

dev = reactant_device(; force=true)

model = Layers.MLP(2, (4, 4, 2), gelu)
shiftto = Layers.ShiftTo(model, ones(Float32, 2), zeros(Float32, 2))

ps, st = Lux.setup(StableRNG(0), shiftto)
ps_ra, st_ra = (ps, st) |> dev

y0, _ = shiftto(st.in_val, ps, st)
@test maximum(abs, y0) < 1.0f-8

x = randn(StableRNG(0), Float32, 2, 2)
x_ra = x |> dev

@test_gradients(
    TestUtils.sumabs2first,
    Constant(shiftto),
    x,
    ps,
    Constant(st);
    atol=1.0f-3,
    rtol=1.0f-3,
    broken_backends=[AutoEnzyme()]
)

st_test = Lux.testmode(st)
st_ra_test = st_test |> dev

@test @jit(shiftto(x_ra, ps_ra, st_ra_test))[1] ≈ shiftto(x, ps, st_test)[1] atol = 1e-3 rtol =
    1e-3

dx_ra, dps_ra = TestUtils.compute_reactant_gradient(shiftto, x_ra, ps_ra, st_ra)
dx_fd, dps_fd = TestUtils.compute_reactant_gradient_fd(shiftto, x_ra, ps_ra, st_ra)

@test dx_ra ≈ dx_fd atol = 1e-3 rtol = 1e-3
@test LuxTestUtils.check_approx(dps_ra, dps_fd; atol=1e-3, rtol=1e-3)
