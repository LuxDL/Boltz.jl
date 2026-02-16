using Boltz, Lux, StableRNGs, Test
using ComponentArrays, DataInterpolations, ForwardDiff

# NOTE: Spline layer tests are CPU-only (no GPU/Reactant support)

@testset "$(spl): train_grid=$(train_grid), dims=$(dims)" for spl in (
        ConstantInterpolation, LinearInterpolation, QuadraticInterpolation, CubicSpline
    ),
    train_grid in (true, false),
    dims in ((), (8,))

    spline = Layers.SplineLayer(dims, 0.0f0, 1.0f0, 0.1f0, spl; train_grid)
    ps, st = Lux.setup(StableRNG(0), spline)

    x = rand(Float32, 4)

    y, st = spline(x, ps, st)
    @test size(y) == (dims..., 4)

    # Test with ComponentArray
    ps_ca = ComponentArray(ps)

    y, st = spline(x, ps_ca, st)
    @test size(y) == (dims..., 4)

    ∂x_fd = ForwardDiff.gradient(x -> sum(abs2, first(spline(x, ps, st))), x)
    ∂ps_fd = ForwardDiff.gradient(ps -> sum(abs2, first(spline(x, ps, st))), ps_ca)

    @test !all(iszero, ∂x_fd) skip = spl === ConstantInterpolation
    @test !all(iszero, ∂ps_fd)
end
