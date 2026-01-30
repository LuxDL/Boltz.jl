using Reactant, Boltz, Lux, StableRNGs, Test, LuxTestUtils, NNlib
using ComponentArrays, ForwardDiff

include("../testutils.jl")

dev = reactant_device(; force=true)

# Test that HNN with wrong output dimensions throws error
hnn_bad = Layers.HamiltonianNN{true}(Layers.MLP(2, (4, 4, 2), NNlib.gelu); autodiff=nothing)
ps, st = Lux.setup(StableRNG(0), hnn_bad)
x = randn(Float32, 2, 4)
@test_throws ArgumentError hnn_bad(x, ps, st)

# Test HNN with correct output dimensions
hnn = Layers.HamiltonianNN{true}(Layers.MLP(2, (4, 4, 1), NNlib.gelu); autodiff=nothing)
ps, st = Lux.setup(StableRNG(0), hnn)
ps_ra, st_ra = (ps, st) |> dev

x = randn(Float32, 2, 4)
x_ra = x |> dev

@test st.first_call
y, st = hnn(x, ps, st)
@test !st.first_call

st_test = Lux.testmode(st)
st_ra_test = st_test |> dev

# Test Reactant forward pass
@test @jit(hnn(x_ra, ps_ra, st_ra_test))[1] ≈ hnn(x, ps, st_test)[1] atol = 1e-3 rtol = 1e-3

dx_ra, dps_ra = TestUtils.compute_reactant_gradient(hnn, x_ra, ps_ra, st_ra)
# TODO: Fix this?? Batching of autodiff calls
# dx_fd, dps_fd = TestUtils.compute_reactant_gradient_fd(hnn, x_ra, ps_ra, st_ra)

# @test dx_ra ≈ dx_fd atol = 1e-3 rtol = 1e-3
# @test LuxTestUtils.check_approx(dps_ra, dps_fd; atol=1e-3, rtol=1e-3)
