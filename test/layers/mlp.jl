using Reactant, Boltz, Lux, StableRNGs, Test, LuxTestUtils

include("../testutils.jl")

dev = reactant_device(; force=true)

act = tanh
norm = (i, ch, act; kwargs...) -> BatchNorm(ch, act; kwargs...)
norm_2 =
    (i, ch, act; kwargs...) ->
        BatchNorm(ch, act; kwargs..., use_decomposed_implementation=true)

model = Layers.MLP(2, (4, 4, 2), act; norm_layer=norm)
model_fd = Layers.MLP(2, (4, 4, 2), act; norm_layer=norm_2)

ps, st = Lux.setup(StableRNG(0), model)
ps_ra, st_ra = (ps, st) |> dev

x = randn(Float32, 2, 2)
x_ra = x |> dev

st_test = Lux.testmode(st)
st_ra_test = st_test |> dev

@test_gradients(
    TestUtils.sumabs2first, Constant(model), x, ps, Constant(st); atol=1e-3, rtol=1e-3,
)

@test @jit(model(x_ra, ps_ra, st_ra_test))[1] ≈ model(x, ps, st_test)[1] atol = 1e-3 rtol =
    1e-3

dx_ra, dps_ra = TestUtils.compute_reactant_gradient(model, x_ra, ps_ra, st_ra)
dx_fd, dps_fd = TestUtils.compute_reactant_gradient_fd(model_fd, x_ra, ps_ra, st_ra)

@test dx_ra ≈ dx_fd atol = 1e-3 rtol = 1e-3
@test LuxTestUtils.check_approx(dps_ra, dps_fd; atol=1e-3, rtol=1e-3)
